from classes import *
import tensorflow as tf

def modelTest():
    epochs = 15
    batch_size = 32
    init_lr = 1e-4
    optimizer = Adam(lr = init_lr, decay = init_lr/epochs)
    metrics = ['accuracy']
    loss = 'binary_crossentropy'
    early_stopping = EarlyStopping(monitor = 'val_accuracy'
                                    ,min_delta = 0
                                    ,patience = 10
                                    ,verbose = 0
                                    ,mode = 'auto')

    myModel = ModelBuilder(epochs
                        ,batch_size
                        ,optimizer
                        ,metrics
                        ,loss
                        ,early_stopping
                        )
def pickleTest():
    dataDir = './data-images'
    myProcessor = DatasetProcessor(dataDir)

    pics, picDirs, camera_model, tampered = myProcessor.dirCollection()

    PickleHandler.save(pics, 'pics')
    PickleHandler.save(picDirs, 'picDirs')
    PickleHandler.save(camera_model, 'camera_model')
    PickleHandler.save(tampered, 'tampered')
    return None
def pickleLoadTest():
    pics = PickleHandler.load('pics')
    picDirs = PickleHandler.load('picDirs')
    camera_model = PickleHandler.load('camera_model')
    tampered = PickleHandler.load('tampered')
    return pics, picDirs, camera_model, tampered

if __name__ == "__main__":

    pics, picDirs, camera_model, tampered = pickleLoadTest()

    #pic = np.array(PIL.Image.open(urllib.request.urlopen(list(df.NEF)[6])))
    #pic = np.array(Image.open(pics[0]))
    plt.imshow(pics[15])
    plt.savefig('test_output/output.png')
    print('done')

    testPic = PRNUProcessor.extract_single(pics[15])
    plt.imshow(testPic)
    plt.savefig('test_output/noiseTest01.png')
    print('noise done')
    quit()


    mnist = tf.keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    myModel = GanBuilder(X_train, Y_train, X_test, Y_test)
    myModel.build_generator()
    myModel.build_discriminator()

    myModel.compile()
    myModel.train()




    quit()
    dataDir = './data-images'
    myProcessor = DatasetProcessor(dataDir)
    print(type(pics[0]))
    resized_pics = DatasetProcessor.resizePics(myProcessor, pics, 100)
    print(resized_pics[0])
    quit()

    df = pd.DataFrame({'dir' : picDirs, 'camera' : camera_model, 'type' : tampered})
    print(df.to_string(index=False))