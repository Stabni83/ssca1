
import sys
import os


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


from pipeline import multiscale_ssca
   


if __name__ == "__main__":
    multiscale_ssca()