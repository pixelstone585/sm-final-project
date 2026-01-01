import numpy as np
from abc import ABC,abstractmethod
from panda3d.core import GraphicsPipe, GraphicsOutput, GraphicsBuffer, Texture, FrameBufferProperties, WindowProperties,GraphicsEngine,CollisionTraverser,CollisionNode,CollisionSphere,CollisionBox,PNMImage, Texture,KeyboardButton,TexturePool,CollisionHandlerQueue
from panda3d.core import LPoint3f, CollisionHandlerPusher

class Obstacle(ABC):

    def __init__(self, dims: np.array =np.array([1,1,1])):
        if (dims.shape != (3,)):
            raise Exception("Dims must be a 3d vector")
        self.scale = dims
        self.rotaion = np.array([0, 0, 0])
        self.position = np.array([0, 0, 0])

        self.collNode = CollisionNode("wall")
        self.collNode.addSolid(CollisionBox(LPoint3f(*(dims / 2)), *(dims / 2)))  # create collider (0,0) at corner

    def setPos(self,pos:np.array):
        self.position=pos

    def _setId(self,id):
        self._id=id
    def getId(self):
        return self._id

    def _setColl(self,coll):
        self._coll=coll

    def _jsonSto(self):
        return{
            "scale": self.dims.tolist(),
            "pos": self.position.tolist(),
            "rot": self.rotaion.tolist(),
            "subname":self.__class__.__name__
        }

    def _jsonLod(self,dict):
        self.scale=np.array(dict["scale"])
        self.rotaion=np.array(dict["rot"])
        self.setPos(np.array(dict["pos"]))

    @abstractmethod
    def onTick(self):
        pass



class mover(Obstacle):
    sign=1
    dims=np.array([1,1,1])
    def onTick(self):
        if(self.position[0] >= 1):
           self.sign=-1
        elif(self.position[0] <= -1):
            self.sign=1
        self.position =self.position+ np.array([0.1, 0, 0])*self.sign
