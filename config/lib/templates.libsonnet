local parser = import "parser.libsonnet";

{
    mbert(language, requires_grad)::
        parser.build_parser(language, "bert",
            ["mbert", "mbert", false, requires_grad]),
    bert_pt(language, epochs, requires_grad)::
        parser.build_parser(language, "bert",
            [language + "_pretrain/epoch_" + epochs,
             "mbert", false, requires_grad]),
    bert_va(language, epochs, requires_grad)::
        parser.build_parser(language, "bert",
            [language + "_augment/epoch_" + epochs,
             "augment", false, requires_grad]),
    bert_dva(language, epochs, requires_grad)::
        parser.build_parser(language, "bert",
            [language + "_dva/epoch_" + epochs,
             "augment", false, requires_grad]),
    elmo(language, epochs)::
        parser.build_parser(language, "elmo",
            ["elmo_" + language + "_" + epochs]),
    ft(language)::
        parser.build_parser(language, "ft", ["trained"]),
}
