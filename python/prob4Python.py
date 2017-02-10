
try:
    import numpy
except ImportError:
    print("numpy is not installed")

nodes = 10

X = open("Input.txt","r").read()
targetOutput = open("Output.txt","r").read()

W_1 = numpy.random.rand(2,nodes)
W_2 = numpy.random.rand(nodes,1)

print(targetOutput)