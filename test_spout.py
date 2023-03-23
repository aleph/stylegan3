# load library
from Library.Spout import Spout

def main() :
    # create spout object
    # spout = Spout(silent = False, width = 1280, height = 720)
    spout = Spout(silent = False, width = 1024, height = 1024)
    # create receiver
    spout.createReceiver('input_gan')
    # create sender
    spout.createSender('output')

    while True :

        # check on close window
        spout.check()
        # receive data
        data = spout.receive()
        # print(data, " | ", data.shape)
        # send data
        spout.send(data)
    
if __name__ == "__main__":
    main()