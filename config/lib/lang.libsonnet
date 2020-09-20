local data_dir = "/m-pinotHD";
// local data_dir = "/data";
{
    "ga": {
        "train_data_path": data_dir + "/echau18/lrlm/ud_links/ga/train.conllu",
        "validation_data_path": data_dir +
                                "/echau18/lrlm/ud_links/ga/dev.conllu",
        "train_size": 858,
    },
    "mt": {
        "train_data_path": data_dir + "/echau18/lrlm/ud_links/mt/train.conllu",
        "validation_data_path": data_dir + "/echau18/lrlm/ud_links/mt/dev.conllu",
        "train_size": 1123,

    },
    "sg": {
        "train_data_path": data_dir + "/echau18/lrlm/ud_links/sg/train.conllu",
        "validation_data_path": data_dir + "/echau18/lrlm/ud_links/sg/dev.conllu",
        "train_size": 2465,
    },
    "vi": {
        "train_data_path": data_dir + "/echau18/lrlm/ud_links/vi/train.conllu",
        "validation_data_path": data_dir + "/echau18/lrlm/ud_links/vi/dev.conllu",
        "train_size": 1400,
    },
}