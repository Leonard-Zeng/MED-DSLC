openclip_backbones = {
    "ViT-B/32": {
        "vision": {"hidden_size": 768,  "ffn_size": 3072, "output_dim": 512, "sequence_length": 50},
        "text":   {"hidden_size": 512,  "ffn_size": 2048, "output_dim": 512, "sequence_length": 77},
    },
    "ViT-B/16": {
        "vision": {"hidden_size": 768,  "ffn_size": 3072, "output_dim": 512, "sequence_length": 197},
        "text":   {"hidden_size": 512,  "ffn_size": 2048, "output_dim": 512, "sequence_length": 77},
    },
    "ViT-L/14": {
        "vision": {"hidden_size": 1024, "ffn_size": 4096, "output_dim": 768, "sequence_length": 257},
        "text":   {"hidden_size": 768,  "ffn_size": 3072, "output_dim": 768, "sequence_length": 77},
    },
    "ViT-H/14": {
        "vision": {"hidden_size": 1280, "ffn_size": None, "output_dim": 1024, "sequence_length": 257},
        "text":   {"hidden_size": 1024, "ffn_size": None, "output_dim": 1024, "sequence_length": 77},
    },
    "ViT-g/14": {
        "vision": {"hidden_size": 1408, "ffn_size": None, "output_dim": 1408, "sequence_length": 257},
        "text":   {"hidden_size": 1408, "ffn_size": None, "output_dim": 1408, "sequence_length": 77},
    },
}
