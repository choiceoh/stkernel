"""dsv41 as a profile package: where its shape derivation lives, and which checkpoint it claims.

The shape wizard discovers profiles instead of listing them (engine/base/kernel_shape.profiles): a model is attached
by adding a directory that declares these two names, and nothing in base learns the model (CHARTER D5).
"""
SHAPES = "engine.profiles.dsv41.shapes"      # the module with kernel_shape_of(ckpt)
MODEL_TYPES = ("deepseek_v41_text",)                     # the text config's model_type this profile derives
