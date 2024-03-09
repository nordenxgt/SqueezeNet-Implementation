from torchinfo import summary

from model import SqueezeNet

def main(): summary(SqueezeNet(), input_size=[1, 3, 224, 224])

if __name__ == "__main__":
    main()