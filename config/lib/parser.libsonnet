local languages = import "lang.libsonnet";
local representations = import "rep.libsonnet";
local NUMPY_SEED = std.extVar("NUMPY_SEED");
local PYTORCH_SEED = std.extVar("PYTORCH_SEED");
local RANDOM_SEED = std.extVar("RANDOM_SEED");
local GRAD_NORM = std.extVar("GRAD_NORM");
local ADAM_BETA_1 = std.extVar("ADAM_BETA_1");
local ADAM_BETA_2 = std.extVar("ADAM_BETA_2");
// for testing purposes only
// local NUMPY_SEED = 13370; 
// local PYTORCH_SEED = 1337;
// local RANDOM_SEED = 133;
// local GRAD_NORM = 1.0;
// local ADAM_BETA_1 = 0.9;
// local ADAM_BETA_2 = 0.99;
{
    build_parser(language, emb_type, params)::
    // BERT params: [model_name, model_vocab, top_layer_only, requires_grad]
    // ELMo params: [model_name]
    // fastText params: [model_name (wiki, cc, or trained)]
    local is_bert_ft = emb_type == "bert" && params[3];
    local batch_size = if is_bert_ft then 
                            (if language == "ga" then 8 else 24)
                            else 64;
    local lang = languages[language];
    local representation_builder = representations[emb_type];
    local representation = representation_builder.build(language, params);
    {
        "numpy_seed": NUMPY_SEED,
        "pytorch_seed": PYTORCH_SEED,
        "random_seed": RANDOM_SEED,
        "dataset_reader": {
            "type": "universal_dependencies",
            "token_indexers": representation["indexers"],
        },
        "train_data_path": lang["train_data_path"],
        "validation_data_path": lang["validation_data_path"],
        "model": {
            "type": "biaffine_parser",
            "arc_representation_dim": 100,
            "tag_representation_dim": 100,
            "use_mst_decoding_for_validation": false,
            "dropout": 0.3,
            "input_dropout": 0.3,
            "text_field_embedder": representation["embedders"],
            "encoder": representation["encoder"],
            // "pos_tag_embedding": {
            //     "embedding_dim": 100,
            //     "sparse": true,
            //     "vocab_namespace": "pos"
            // },
            "initializer": [
                [".*projection.*weight", {"type": "xavier_uniform"}],
                [".*projection.*bias", {"type": "zero"}],
                [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
                [".*tag_bilinear.*bias", {"type": "zero"}],
                [".*weight_ih.*", {"type": "xavier_uniform"}],
                [".*weight_hh.*", {"type": "orthogonal"}],
                [".*bias_ih.*", {"type": "zero"}],
                [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
            ],
        },
        "iterator": {
            "type": "bucket",
            "sorting_keys": [["words", "num_tokens"]],
            "batch_size": batch_size, 
        },
        "trainer": {
            "num_epochs": 200,
            "grad_norm": GRAD_NORM,
            "patience": 20,
            "cuda_device": 0,
            "validation_metric": "+LAS",
            "should_log_learning_rate": true,
            "should_log_parameter_statistics": true,
            // the only difference between DSAdam and BERT AdamW is the weight
            // decay, grad clipping, and the lack of bias correction.
            // However, DSAdam supports sparse gradients out of the box, and
            // HuggingFace finds them similar
            // (https://github.com/huggingface/transformers/issues/420)
            // If we get rid of POS features in the future, though, we should
            // be able to directly use the "bert_adam" optimizer below.
            "optimizer": {
                "type": "dense_sparse_adam",
                "betas": [ADAM_BETA_1, ADAM_BETA_2],
                // faster LR; the slower one will be computed via decay_factor
                // in the scheduler
                "lr": 1e-3, 
                "parameter_groups": [
                    [
                        ["^text_field_embedder.*.bert_model.embeddings",
                         "^text_field_embedder.*.bert_model.encoder"],
                        {}
                    ],
                    [
                        ["^text_field_embedder.*._scalar_mix",
                         "^text_field_embedder.*.pooler",
                         "^_head_sentinel",
                         "^encoder",
                         "^head_arc_feedforward",
                         "^child_arc_feedforward",
                         "^arc_attention",
                         "^head_tag_feedforward",
                         "^child_tag_feedforward",
                         "^tag_bilinear",
                         "^_pos_tag_embedding"],
                         {}
                    ]
                ],
            },
            // "optimizer": {
            //     "type": "bert_adam",
            //     "b1": ADAM_BETA_1,
            //     "b2": ADAM_BETA_2,
            //     "weight_decay": 0.01,
            //     // faster LR; the slower one will be computed via decay_factor
            //     // in the scheduler
            //     "lr": 1e-3, 
            //     "parameter_groups": [
            //         [
            //             ["^text_field_embedder.*.bert_model.embeddings",
            //              "^text_field_embedder.*.bert_model.encoder"],
            //             {}
            //         ],
            //         [
            //             ["^text_field_embedder.*._scalar_mix",
            //              "^text_field_embedder.*.pooler",
            //              "^_head_sentinel",
            //              "^encoder",
            //              "^head_arc_feedforward",
            //              "^child_arc_feedforward",
            //              "^arc_attention",
            //              "^head_tag_feedforward",
            //              "^child_tag_feedforward",
            //              "^tag_bilinear",
            //              "^_pos_tag_embedding"],
            //              {}
            //         ]
            //     ],
            //     // warmup, t_total, and schedule kept to default because we
            //     // use a learning rate schedule
            //     // e (epsilon) also kept default
            //     // max_grad_norm kept default
            // },
            "learning_rate_scheduler": if is_bert_ft then {
                "type": "ulmfit_sqrt",
                "model_size": 1, // UDify did this so...?
                // language-specific one epoch
                // TODO: check to see that this calculation is correct
                "warmup_steps": std.ceil(lang["train_size"] / batch_size),
                // language-specific one epoch, by suggestion of UDify
                // https://github.com/Hyperparticle/udify/issues/6
                "start_step": std.ceil(lang["train_size"] / batch_size),
                "factor": 5.0, // following UDify
                "gradual_unfreezing": true,
                "discriminative_fine_tuning": true,
                "decay_factor": 0.05, // yields a slow LR of 5e-5
                // steepness kept to 0.5 (sqrt)
            },
        },
    },    
}
