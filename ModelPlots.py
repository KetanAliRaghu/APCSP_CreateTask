from keras.utils import plot_model
from keras.saving.save import load_model

model = load_model('models/TFChatbot_99Acc_Mem')
plot_model(model, 'model_plots/TFChatbot_99Acc_Mem_Plot.png',
           show_shapes = True,
           show_dtype = True,
           show_layer_names = True,
           expand_nested = True,
           show_layer_activations = True)

model = load_model('models/GenerationPipelineE_20Epoch')
plot_model(model, 'model_plots/GenerationPipelineE_20Epoch_Plot.png',
           show_shapes = True,
           show_dtype = True,
           show_layer_names = True,
           expand_nested = True,
           show_layer_activations = True)

model = load_model('models/EmDetModel_4Class_20Epoch_88Acc')
plot_model(model, 'model_plots/EmDetModel_4Class_20Epoch_88Acc_Plot.png',
           show_shapes = True,
           show_dtype = True,
           show_layer_names = True,
           expand_nested = True,
           show_layer_activations = True)
