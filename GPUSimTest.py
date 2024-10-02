import pyopencl
import numpy
import time
"""
Creates a square grid with the dimensions entered at runtime
stores it in a buffer
runs the simulation on the GPU as fast as it can go
every 1000 cycles, it will print the average cycles per second and the average cells simulated per second
"""


#create a context
context=pyopencl.create_some_context()
q=pyopencl.CommandQueue(context)
memFlags=pyopencl.mem_flags
class GOLChunkSimulator():
    def __init__(self,context,queue,cw,ch):
        """
        Simulator for Conway's Game of Life on world chunks of arbitrary size.
        The world chunk is represented as an array of int32's (may change later for memory efficiency)
        in the array, there is a border of cells all the way around the actual useable portion
        that represents the borders of neighboring chunks. This border will not have the simulation run on it.
        Therefore, to have a simulated chunk size of 8X8, you must choose a chunk size of 10X10 in the constructor.

        cw      : chunk width
        ch      : chunk height
        context : opencl context
        queue   : opencl queue
        """
        self.context=context
        self.queue=queue
        self.w=cw
        self.h=ch
        self.outArray=numpy.empty(shape=cw*ch, dtype=numpy.dtype("i"))
        self.startBuffer=pyopencl.Buffer(context,memFlags.READ_ONLY,self.outArray.nbytes)
        
        self.endBuffer=pyopencl.Buffer(context, memFlags.WRITE_ONLY, self.outArray.nbytes)
        self.prg=pyopencl.Program(context,
"""
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
__kernel void run(
    __global const int *prev, __global int *res, int const w, int const h)
{
  int gid=get_global_id(0);
  int gidym=MAX(gid-w,1);
  int gidyp=MIN(gid+w+1,w*h);
  int n=prev[gid-1]+prev[gid+1]+prev[gidym-1]+prev[gidym]+prev[gidym+1]+prev[gidyp]+prev[gidyp-1]+prev[gidyp-2];
  res[gid]= n==3 || (n==2&&prev[gid]);
}
"""#n==3 || (n==2&&prev[gid]);
        ).build()
    def setIn (self,array):
        """Sets the input chunk for the simulation."""
        pyopencl.enqueue_copy(self.queue,self.startBuffer,array)
    def getOut (self):
        """
        Copies result of simulation into an array and returns that array.
        """
        pyopencl.enqueue_copy(self.queue,self.outArray,self.endBuffer)
        return self.outArray
    def createEmpty(self):
        """
        Returns an empty chunk array with the proper size that is filled with zeros.
        """
        return numpy.zeros_like(self.outArray)
    def run (self):
        """
        runs GOL simulation. To get the results, call "getOut()"
        """
        self.prg.run(
            self.queue,
            self.outArray.shape,
            None,
            self.startBuffer,
            self.endBuffer,
            numpy.int32(self.w),
            numpy.int32(self.h)
        )
    def checkbounds(self, chunkArray):
        """
        checks edges of chunk [internal chunk, not counting the fringe]
        and returns a touple that contains a boolean for if there is a live cell
        in any of the cells on a side. Gives them in this order:

        (top, right, bottom, left)

        or clockwise around the chunk starting from the top.
        """
        return(
            numpy.any(chunkArray[self.w+1:self.w*2-1]),#top
            numpy.any(chunkArray[self.w*2-2:(self.w*(self.h))-2:self.w]),#right
            numpy.any(chunkArray[(self.w*(self.h-2))+1:(self.w*(self.h-1))-1]),#bottom
            numpy.any(chunkArray[self.w+1:(self.w*(self.h-1))+1:self.w])#left
        )


size=int(input("choose chunk size"))
sim=GOLChunkSimulator(context,q,size,size)
chunks={
    "0,0":sim.createEmpty(),
}
y=12
x=12
c=x+y*size

"""chunks["0,0"][c]=1
chunks["0,0"][c-size-1]=1
chunks["0,0"][c-size+1]=1
"""
chunks["0,0"][c-size]=1
chunks["0,0"][c+2]=1
chunks["0,0"][c+size]=1
chunks["0,0"][c+size-1]=1
chunks["0,0"][c+size+1]=1



"""
    for x in range(0,size):
        print(chunks["0,0"][x*size:x*size+size])
    print("\n")
    time.sleep(0.5)
"""
st=time.time_ns()
cnt=0
while 1:
    #sim.setIn(chunks["0,0"])
    sim.run()
    #chunks["0,0"]=sim.getOut()
    cnt+=1
    if(cnt==1000):
        nd=time.time_ns()
        dt=nd-st
        fps=1000*(1000000000)/dt
        print(str(fps)+"@"+str(fps*size*size))
        st=nd
        cnt=0
