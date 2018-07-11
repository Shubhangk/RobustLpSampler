import sys

def main(m, p, epochs, eps, stream_size, stream):
    print(m)
    i = int(m)
    p = float(p)
    dataset = [int(j) for j in range(len(stream))]



if __name__ == '__main__':
    print(sys.argv)
    stream = sys.argv[5:]
    stream = [int(stream[i]) for i in range(len(stream))]
    print(stream)
    #main(sys.argv[1:])
