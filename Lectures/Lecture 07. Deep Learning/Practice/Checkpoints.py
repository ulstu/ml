# it is no fully working code
# but it's example of checkpoint saving
# neural weights in training process.
# Is important because train is long process
# and getting reserve copy of model weights
# allow to time economy

model = GetModel(mode='load_W', filename='weights-improvement-14-0.3587.hdf5', X=X, Y=Y)

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, Y, nb_epoch=epoch, batch_size=64, verbose=1, shuffle=True, callbacks=callbacks_list)