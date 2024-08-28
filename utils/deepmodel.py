import tensorflow as tf

def deepmodel(topology, predictand, inputShape, outputShape):
	### deepesd
	if topology == 'deepesd' or 'noAddVal-deepesd' or 'ta-deepesd' or 'anom-deepesd':
		l0 = tf.keras.Input(shape = inputShape)
		l1 = tf.keras.layers.Conv2D(50,[3,3], activation = 'relu')(l0)
		l2 = tf.keras.layers.Conv2D(25,[3,3], activation = 'relu')(l1)
		l3 = tf.keras.layers.Conv2D(10,[3,3], activation = 'relu')(l2)
		l4 = tf.keras.layers.Flatten()(l3)
		if predictand == 'tas':
			l5 = tf.keras.layers.Dense(outputShape)(l4)
		elif predictand == 'RAINNC':
			l51 = tf.keras.layers.Dense(outputShape, activation = "sigmoid")(l4)
			l52 = tf.keras.layers.Dense(outputShape)(l4)
			l53 = tf.keras.layers.Dense(outputShape)(l4)
			l5 = tf.stack([l51,l52,l53], axis = 2)
		model = tf.keras.Model([l0],[l5])
	### unet
	if topology == 'unet':
		l0 = tf.keras.Input(shape = inputShape)
		l1 = tf.keras.layers.Conv2D(64,[3,6], activation = 'relu', padding = 'valid')(l0) # batch normalization entre convolucion y activacion
		l2 = tf.keras.layers.MaxPooling2D()(l1)
		l3 = tf.keras.layers.Conv2D(128,[2,2], activation = 'relu', padding = 'same')(l2)
		l4 = tf.keras.layers.MaxPooling2D()(l3)
		l5 = tf.keras.layers.Conv2D(256,[1,1], activation = 'relu', padding = 'same')(l4)
		l6 = tf.keras.layers.MaxPooling2D()(l5)
		l7 = tf.keras.layers.Conv2DTranspose(256,[2,2], strides = [2,2])(l6)
		l8 = tf.keras.layers.concatenate([l5,l7])
		l9 = tf.keras.layers.Conv2D(256,[2,2], activation = 'relu', padding = 'same')(l8)
		l10 = tf.keras.layers.Conv2DTranspose(128,[2,2], strides = [2,2])(l9)
		l11 = tf.keras.layers.concatenate([l3,l10])
		l12 = tf.keras.layers.Conv2D(128,[2,2], activation = 'relu', padding = 'same')(l11)
		l13 = tf.keras.layers.Conv2DTranspose(64,[2,2], strides = [2,2])(l12)
		l14 = tf.keras.layers.concatenate([l1,l13])
		l15 = tf.keras.layers.Conv2D(64,[2,2], activation = 'relu', padding = 'same')(l14)
		l16 = tf.keras.layers.Conv2DTranspose(64,[2,2], strides = [2,2])(l15)
		l17 = tf.keras.layers.Conv2D(64,[2,2], activation = 'relu', padding = 'same')(l16)
		l18 = tf.keras.layers.Conv2DTranspose(64,[2,2], strides = [2,2])(l17)
		l19 = tf.keras.layers.Conv2D(64,[2,2], activation = 'relu', padding = 'same')(l18)
		l20 = tf.keras.layers.Conv2DTranspose(64,[2,2], strides = [2,2])(l19)
		l21 = tf.keras.layers.Conv2D(64,[2,2], activation = 'relu', padding = 'same')(l20)
		l22 = tf.keras.layers.Conv2DTranspose(64,[2,2], strides = [2,2])(l21)
		l23 = tf.keras.layers.Conv2D(64,[2,2], activation = 'relu', padding = 'same')(l22)
		l24 = tf.keras.layers.Conv2D(64,[6,2], activation = 'relu', padding = 'valid')(l23)
		l25 = tf.keras.layers.Conv2D(64,[6,2], activation = 'relu', padding = 'valid')(l24)
		l26 = tf.keras.layers.Conv2D(64,[6,2], activation = 'relu', padding = 'valid')(l25)
		l27 = tf.keras.layers.Conv2D(64,[6,2], activation = 'relu', padding = 'valid')(l26)
		l28 = tf.keras.layers.Conv2D(64,[8,4], activation = 'relu', padding = 'valid')(l27)
		l29 = tf.keras.layers.Conv2D(64,[1,1], activation = 'relu', padding = 'same')(l28)
		l30 = tf.keras.layers.Conv2D(1,[1,1], activation = 'linear', padding = 'same')(l29)
		model = tf.keras.Model([l0],[l30])
	return model
