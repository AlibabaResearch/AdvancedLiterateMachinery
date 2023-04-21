from model.geolayoutlm_vie import GeoLayoutLMVIEModel

def get_model(cfg):
    if cfg.model.head in ["vie"]:
        model = GeoLayoutLMVIEModel(cfg=cfg)
    else:
        raise ValueError(f"Unknown cfg.model.head={cfg.model.head}")

    return model
