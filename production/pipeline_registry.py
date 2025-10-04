PIPELINE_REGISTRY = {}



# internally this is how it works
# @register_pipeline("pipeline_name")
# def pipeline_function():
#     pass
# pipeline_function = register_pipeline("pipeline_name")(pipeline_function)
# pipeline_function()


def register_pipeline(pipeline_name):
    def decorator(func):
        PIPELINE_REGISTRY[pipeline_name] = func
        return func
    return decorator



