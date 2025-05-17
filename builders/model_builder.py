from .register import Registry

META_ARCHITECTURE = Registry(name="ARCHITECTURE")

def build_model(config):
    model = META_ARCHITECTURE.get(config.model.architecture)(config)
    
    return model
