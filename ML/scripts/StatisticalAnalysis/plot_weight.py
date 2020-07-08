from keras.models import load_model



def Mean_Squared_over_true_Error(y_true, y_pred):
# Create a custom loss function that divides the difference by the true
#if not K.is_tensor(y_pred):
if not K.is_keras_tensor(y_pred):
    y_pred = K.constant(y_pred)

y_true = K.cast(y_true, y_pred.dtype)
diff_ratio = K.square((y_pred - y_true)/K.clip(K.abs(y_true),K.epsilon(),None))
loss = K.mean(diff_ratio, axis=-1)
# Return a function

return loss

data_path = "/pylon5/as5phnp/tbilling/sandbox/hyper_param_optimiz/ml_paper1/add_conv_layer/v1/"
results = sorted(glob.glob(data_path+"model_fold*.h5"))

model = load_model(outputdir+'hyperParam_model.h5',
custom_objects={"Mean_Squared_over_true_Error":Mean_Squared_over_true_Error})

"""
    [<keras.layers.convolutional.Conv2D at 0x7f9f59e73cc0>,
    <keras.layers.normalization.BatchNormalization at 0x7f9f59e73208>,
    <keras.layers.pooling.MaxPooling2D at 0x7f9f59e73dd8>,
    <keras.layers.convolutional.Conv2D at 0x7f9f59e73fd0>,
    <keras.layers.normalization.BatchNormalization at 0x7f9f59ed5c50>,
    <keras.layers.pooling.MaxPooling2D at 0x7f9f59ebbf28>,
    <keras.layers.convolutional.Conv2D at 0x7f8153a74748>,
    <keras.layers.normalization.BatchNormalization at 0x7f8153a085c0>,
    <keras.layers.convolutional.Conv2D at 0x7f81539edeb8>,
    <keras.layers.normalization.BatchNormalization at 0x7f81539b3f28>,
    <keras.layers.pooling.MaxPooling2D at 0x7f8153975a20>,
    <keras.layers.pooling.GlobalAveragePooling2D at 0x7f8153884668>,
    <keras.layers.core.Dropout at 0x7f81538994e0>,
    <keras.layers.core.Dense at 0x7f81538adf98>,
    <keras.layers.core.Dropout at 0x7f815383fe10>,
    <keras.layers.core.Dense at 0x7f815383fcc0>,
    <keras.layers.core.Dropout at 0x7f81537fda20>,
    <keras.layers.core.Dense at 0x7f81537fd898>,
    <keras.layers.core.Dense at 0x7f8153744860>]
"""

# print type of layer
print(type(model.layers[0]))

model.layers[1].get_weights()[1]

plt.figure(figsize=(10,8))

plt.xlabel('True')
plt.xlim(0.028,0.1)
#plt.xscale('log')

plt.ylabel('Predicted')
plt.legend(markerscale=2.5)

filename = outputdir+'PredvsTruth{}_{:d}.png'.format(steps,fold)
plt.savefig(filename)
plt.clf()
