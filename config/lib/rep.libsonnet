local mbert_path = "/m-pinotHD/echau18/bert/mbert";
local outputs = "/m-pinotHD/echau18/bert/outputs";
local bert_data = "/m-pinotHD/echau18/bert/data";
local main_fasttext_path = "/m-pinotHD/echau18/lrlm/fasttext_links";

local default_encoder(dim) = {
    "type": "stacked_bidirectional_lstm",
    "hidden_size": 400,
    "input_size": dim,
    "num_layers": 3,
    "recurrent_dropout_probability": 0.3,
    "use_highway": true
};

local passthrough_encoder(dim) = {
    "type": "pass_through",
    "input_dim": dim,
};

{
    "bert": {
        build(language, params)::
        // BERT params: [model_name, model_vocab, top_layer_only, requires_grad]
        // model name: `mbert` or a model-epoch name, e.g., `sg-augment/1`
        // model vocab: `mbert` or `augment`
        local model_name = if params[0] == "mbert" then mbert_path
                            else outputs + "/" + params[0];
        local model_vocab = if params[1] == "mbert" then mbert_path +
                            "/vocab.txt" 
                            else if params[1] == "augment" then
                            bert_data + "/" + language
                            + "/vocab-augmentation/vocab-mbert-5000.txt"
                            else error "Unknown vocab.";
        local top_layer_only = params[2];
        local requires_grad = params[3];
        local dim = 768;
        {
            "dim": dim,
            "embedders": {
                "allow_unmatched_keys": true,
                "embedder_to_indexer_map": {
                    "bert": ["bert", "bert-offsets"],
                },
                "token_embedders": {
                    "bert": {
                        "type": "bert-pretrained-with-dropout",
                        "pretrained_model": model_name,
                        "top_layer_only": top_layer_only,
                        "requires_grad": requires_grad,
                        "layer_dropout": 0.1,
                    },
                },
            },
            "encoder": if requires_grad then passthrough_encoder(dim)
                        else default_encoder(dim),
            "indexers": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": model_vocab,
                    // we're using cased models
                    "do_lowercase": false,
                    "truncate_long_sequences": false,
                    "use_starting_offsets": true,
                },
            },
        },
    },
    "elmo": {
        build(language, params)::
        // ELMo params: [model_name]
        local elmo_model = params[0];
        local dim = 1024;
        {
            "dim": dim,
            "embedders": {
                "elmo": {
                    "type": "elmo_token_embedder_variable",
                    "do_layer_norm": true,
                    "dropout": 0.5,
                    "requires_grad": true,
                    "char_map_file": outputs + "/" + elmo_model + "/char_vocab.txt",
                    "options_file": outputs + "/" + elmo_model +
                                    "/options.json",
                    "weight_file": outputs + "/" + elmo_model + "/weights.hdf5",
                },
            },
            "encoder": default_encoder(dim),
            "indexers": {
                "elmo": {
                    "type": "elmo_characters_variable",
                    "char_map_file": outputs + "/" + elmo_model + "/char_vocab.txt",
                },
            },
        },
    },
    "ft": {
        build(language, params)::
        // fastText params: [model_name (wiki, cc, or trained)]
        local fasttext_model = params[0];
        local suffix = if fasttext_model == "trained" then "model.vec"
                       else if fasttext_model == "wiki" then "wiki.vec"
                       else if fasttext_model == "cc" then "cc.vec.gz"
                       else error "Unknown fastText model.";
        local emb_dim = if fasttext_model == "trained" then 100 else 300;
        local path = if fasttext_model == "trained" then
                        bert_data + "/" + language + "/unlabeled/fasttext/" +
                            suffix
                        else main_fasttext_path + "/" + language + "/" + suffix;
        {
            "dim": emb_dim,
            "embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": emb_dim,
                    "pretrained_file": path,
                    "sparse": true,
                    "trainable": true,
                },
            },
            "encoder": default_encoder(emb_dim),
            "indexers": {
                "tokens": {
                    "type": "single_id",
                },
            },
        },
    },
}
