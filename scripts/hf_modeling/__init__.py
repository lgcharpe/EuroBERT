from .configuration_optimus import OptimusConfig

from .modeling_optimus import (
    OptimusForMLM,
    OptimusForSequenceClassification,
    OptimusForQuestionAnswering,
    OptimusForTokenClassification,
    OptimusModel,
)

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)

AutoConfig.register("optimus", OptimusConfig)
AutoModel.register(OptimusConfig, OptimusModel)
AutoModelForMaskedLM.register(OptimusConfig, OptimusForMLM)
AutoModelForSequenceClassification.register(
    OptimusConfig, OptimusForSequenceClassification
)
AutoModelForQuestionAnswering.register(OptimusConfig, OptimusForQuestionAnswering)
AutoModelForTokenClassification.register(OptimusConfig, OptimusForTokenClassification)


# register optimus model type
OptimusConfig.register_for_auto_class()
OptimusModel.register_for_auto_class("AutoModel")
OptimusForMLM.register_for_auto_class("AutoModelForMaskedLM")
OptimusForSequenceClassification.register_for_auto_class(
    "AutoModelForSequenceClassification"
)
OptimusForQuestionAnswering.register_for_auto_class("AutoModelForQuestionAnswering")
OptimusForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
