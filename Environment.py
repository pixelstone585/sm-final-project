from direct.showbase.ShowBase import ShowBase
from panda3d.core import LPoint3f, CollisionHandlerPusher
from direct.actor.Actor import Actor
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import GraphicsPipe, GraphicsOutput, GraphicsBuffer, Texture, FrameBufferProperties, WindowProperties,GraphicsEngine,CollisionTraverser,CollisionNode,CollisionSphere,CollisionBox,PNMImage, Texture,KeyboardButton,TexturePool,CollisionHandlerQueue

from direct.showbase.Loader import Loader
from direct.gui.DirectEntry import DirectEntry

import numpy as np
print("G")
from PIL import Image

import time

import matplotlib.pyplot as plt

import math
import json
import sys
import copy

from abc import ABC,abstractmethod

from Obstacles import *

class Drone:

	position=np.array([0,0,0])
	rotation=np.array([0,0,0])
	#actor=Actor("models/panda")
	actor=Actor()

	def __init__(self):
		self.actor.setScale(0.1,0.1,0.1)

		self.colliderNode=CollisionNode("drone")
		self.colliderNode.addSolid(CollisionBox(LPoint3f(0,0,0),8,8,8))
		self.collider=self.actor.attachNewNode(self.colliderNode)
		#self.collider.show()

	def rotate(self,dRot : np.array):
		if(dRot.shape!=(3,)):
			raise Exception("dRot must be a 3d vector")
		self.rotation =self.rotation+dRot

	def move_old(self,dPos : np.array):
		if(dPos.shape!=(3,)):
			raise Exception("dPos must be a 3d vector")
		self.position=np.array([*self.actor.getPos()])+dPos
		self.actor.setPos(*self.position)

	def move(self,dPos : np.array):
		if (dPos.shape != (3,)):
			raise Exception("dPos must be a 3d vector")

		#this was a pain to figure out

		#forward - backword
		y=dPos[1]*math.cos(math.radians(self.rotation[1]))*math.cos(math.radians(self.rotation[0]))
		x=-dPos[1]*math.cos(math.radians(self.rotation[1]))*math.sin(math.radians(self.rotation[0]))
		z=dPos[1]*math.sin(math.radians(self.rotation[1]))

		#left - right
		y+=dPos[0]*math.cos(math.radians(self.rotation[1]))*math.sin(math.radians(self.rotation[0]))
		x+=dPos[0]*math.cos(math.radians(self.rotation[1]))*math.cos(math.radians(self.rotation[0]))*math.cos(math.radians(self.rotation[2]))
		z+=-dPos[0]*math.sin(math.radians(self.rotation[2]))

		#up - down
		y+=-dPos[2]*math.sin(math.radians(self.rotation[1]))
		x+=dPos[2]*math.sin(math.radians(self.rotation[2]))
		z+=dPos[2]*math.cos(math.radians(self.rotation[1]))*math.cos(math.radians(self.rotation[0]))*math.cos(math.radians(self.rotation[2]))

		self.position = np.array([*self.actor.getPos()]) + np.array([x,y,z])
		self.actor.setPos(*self.position)

	def sync(self):
		self.position = np.array([*self.actor.getPos()]) # sync in case of collision

	def setPos(self,pos : np.array):
		if(pos.shape!=(3,)):
			raise Exception("Pos must be a 3d vector")
		self.position=pos
		self.actor.setPos(*pos)

	def setRot(self,rot : np.array):
		if(rot.shape!=(3,)):
			raise Exception("rot must be a 3d vector")

		self.rotation=rot



class Wall():
	def __init__(self, dims : np.array):
		if(dims.shape != (3,)):
			raise Exception("Dims must be a 3d vector")
		self.scale=dims
		self.rotaion=np.array([0,0,0])
		self.position=np.array([0.0,0.0,0.0])

		self.collNode=CollisionNode("wall")
		self.collNode.addSolid(CollisionBox(LPoint3f(*(dims/2)),*(dims/2)))#create collider (0,0) at corner
	def setPos(self,pos:np.array):
		self.position=pos

	def _setId(self,id):
		self._id=id
	def getId(self):
		return self._id
	def _setColl(self,coll):
		self._coll=coll

	def _jsonSto(self):
		return {
			"scale": self.scale.tolist(),
			"pos": self.position.tolist(),
			"rot":self.rotaion.tolist()
		}
	def _jsonLod(self,dict):
		self.setPos(np.array(dict["pos"]))
		self.scale=np.array(dict["scale"])
		self.rotaion=np.array(dict["rot"])


class Module:
	def __init__(self):
		self.obstacles=[]
		self.walls=[]
		self.offset=np.array([0.0,0.0,0.0])

	def setOffset(self,offset :np.array):
		self.offset=offset

	def getOffset(self):
		return self.offset

	def addObstacle(self, obst : Obstacle):
		self.obstacles.append(obst)

	def addWall(self, wall : Wall):
		self.walls.append(wall)

	def _formatJson(self):
		dictsObs=[]
		for x in self.obstacles:
			if not x is None:
				dictsObs.append(x._jsonSto())

		dictsWall = []
		for x in self.walls:
			if not x is None:
				dictsWall.append(x._jsonSto())
		return {
			"obs":dictsObs,
			"walls":dictsWall,
			"offset":self.offset.tolist()
		}

	def _unpackJson(self, dict):
		dictwalls=dict["walls"]
		for x in dictwalls:
			self.walls.append(Wall(np.array([1,1,1])))
			self.walls[-1]._jsonLod(x)

		dictobs=dict["obs"]
		for x in dictobs:
			subclass=getattr(__import__("Obstacles"),x["subname"])
			self.obstacles.append(subclass())
			self.obstacles[-1]._jsonLod(x)

		self.offset=np.array(dict["offset"])

class Scene:
	def __init__(self):
		self.walls=[]
		self.obstacles=[]
		self.startPos=np.array([0,0,0])
		self.startRot=np.array([0,0,0])
		self.goal=np.array([0,0,0])

	def addWall(self,wall : Wall):
		self.walls.append(wall)
	def addWalls(self,walls):
		self.walls+=walls

	def addObstacle(self,obstacle : Obstacle):
		self.obstacles.append(obstacle)

	def addObstacles(self,obstacles):
		self.obstacles+=obstacles

	def setStartPos(self,pos : np.array):
		self.startPos=pos

	def setStartRot(self,rot :np.array):
		self.startRot=rot

	def setGoal(self,pos : np.array):
		self.goal=pos

	def addModule(self, module:Module ):
		for x in module.walls:
			self.walls.append(copy.deepcopy(x))
			self.walls[-1].position=self.walls[-1].position+module.getOffset()
		for x in module.obstacles:
			self.obstacles.append(copy.deepcopy(x))
			self.obstacles[-1].position=self.obstacles[-1].position+module.getOffset()

	def saveAsModule(self):
		tmpmod=Module()
		for wall in self.walls:
			tmpmod.addWall(wall)
		print("saved!")
		return tmpmod._formatJson()






class Engine(ShowBase):
	def __init__(self,drone : Drone, sensor_range=50,debugCam=False):
		ShowBase.__init__(self)
		self.walls=[]
		self.obstacles=[]
		self.obstaclesObj=[]

		self.test_wall=Wall(np.array([1,1,1]))

		self.disp=OnscreenText(text="dist: 0", pos=(-1.3, 0.6), scale=0.1, fg=(1, 1, 1, 1), align=0, mayChange=True)


		#jank depth buffer extraction setup
		winprops = WindowProperties.size(self.win.getXSize(), self.win.getYSize())
		fbprops = FrameBufferProperties()
		fbprops.setDepthBits(1)
		self.depthBuffer = self.graphicsEngine.makeOutput(
			self.pipe, "depth buffer", -2,
			fbprops, winprops,
			GraphicsPipe.BFRefuseWindow,
			self.win.getGsg(), self.win)
		self.depthTex = Texture()
		self.depthTex.setFormat(Texture.FDepthComponent)
		self.depthBuffer.addRenderTexture(self.depthTex,
										  GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
		lens = self.cam.node().getLens()
		lens.setFar(sensor_range)
		self.depthCam = self.makeCamera(self.depthBuffer,lens=lens,scene=self.render)
		self.depthCam.reparentTo(self.cam)

		# setup collision
		self.cTrav = CollisionTraverser()
		self.pusher = CollisionHandlerPusher()
		self.coll_queue = CollisionHandlerQueue()

		#setup drone
		self.debugCam=debugCam

		self.drone=drone
		self.pusher.addCollider(drone.collider, drone.actor)
		self.cTrav.addCollider(drone.collider, self.pusher)

		#set up collision callback
		self.pusher.addInPattern("%fn-into-%in")
		self.accept("drone-into-wall",self.raiseCollFlag)
		self.collFlag=False



		self.colliders=[]

		drone.actor.reparentTo(self.render)
		#drone.actor.detachNode()

		self.goal=np.array([0,0,0])

		img = PNMImage(1, 1, 1)
		img.fill(0.7, 0.7, 0.7)
		#self.white_tex = Texture("white")
		#self.white_tex.load(img)
		self.white_tex=TexturePool.loadTexture(r"tmp_img.png")

		self.devInput = DirectEntry(
			initialText="Enter wall dims (speprated by commas):",
			numLines=1,
			width=25,
			scale=0.07,
			pos=(-1, 0, 0),
			command=self.devAdd,
			#extraArgs=[self.myEntry]
		)
		self.devInputPos = DirectEntry(
			initialText="Enter wall pos (speprated by commas):",
			numLines=1,
			width=25,
			scale=0.07,
			pos=(-1, 0, 0),
			command=self.devSetPos,
			# extraArgs=[self.myEntry]
		)
		self.devPos=np.array([0,0,0])
		self.devInput.hide()
		self.devInputPos.hide()

	#workaround for jank camera parenting
	#x,z,y
	def updateCam(self,offset=np.array([0,-0.5,0])):
		if not self.debugCam:
			#offset: np.array([0,-0.7,0])
			#self.cam.setPos(*(self.drone.position+np.array([-1,-1,-1])))
			tmp=self.loader.loadModel(modelPath="models/box")
			tmp.setPos(*(self.drone.position)+offset)

			self.cam.setPos(*(self.drone.position+offset))
			self.cam.setHpr(*self.drone.rotation)

	def addWall(self,wall : Wall,showCollider =False):
		self.walls.append(self.loader.loadModel(modelPath="models/box"))
		self.walls[-1].setScale(*wall.scale)
		self.walls[-1].setPos(*wall.position)
		self.walls[-1].setHpr(*wall.rotaion)
		self.walls[-1].reparentTo(self.render)
		#coloring
		self.walls[-1].setTexture(self.white_tex,1)
		#store position in wall object
		wall._setId(len(self.walls)-1)
		coll=self.render.attachNewNode(wall.collNode)
		coll.setPos(*wall.position)
		wall._setColl(coll)
		self.colliders.append(coll)
		if showCollider:
			coll.show()

	def addObstacle(self,obstacle : Obstacle,showCollider=False):
		self.obstaclesObj.append(obstacle)
		self.obstacles.append(self.loader.loadModel(modelPath="models/box"))
		self.obstacles[-1].setScale(*obstacle.scale)
		self.obstacles[-1].setPos(*obstacle.position)
		self.obstacles[-1].setHpr(*obstacle.rotaion)
		self.obstacles[-1].reparentTo(self.render)
		# coloring
		self.obstacles[-1].setTexture(self.white_tex, 1)
		self.obstacles[-1].setColor(1,0,0,1)
		# store position in wall object
		obstacle._setId(len(self.obstacles) - 1)
		coll = self.render.attachNewNode(obstacle.collNode)
		coll.setPos(*obstacle.position)
		obstacle._setColl(coll)
		self.colliders.append(coll)
		if showCollider:
			coll.show()

	def addGoal(self,pos : np.array,doesRender=False):
		self.goal=pos
		"""self.goalRender=self.loader.loadModel(modelPath="models/box")
		self.goalRender.setPos(0,pos[1],0)
		self.goalRender.setScale(10,1,10)
		if(doesRender):
			self.goalRender.reparentTo(self.render)
		else:
			self.goalRender.detachNode()

		#self.goalRender.setTexture(self.white_tex,1)
		#self.goalRender.setColor(0,1,0,1)"""

	def setGoalRender(self,toggle):
		if(toggle):
			self.goalRender.reparentTo(self.render)
		else:
			self.goalRender.detachNode()
	def addRuler(self,length):
		self.ruler=self.loader.loadModel(modelPath="models/box")
		self.ruler.setPos(0,length/2,0)
		self.ruler.setScale(1,length,1)
		self.ruler.reparentTo(self.render)
		self.ruler.setColor(0,0.82745098039,0.98823529411,1)



	def tick(self):
		self.drone.sync()
		#drone.move(np.array([0,0.1,0]))
		for x in self.obstaclesObj:
			x.onTick()
			self.obstacles[x._id].setScale(*x.scale)
			self.obstacles[x._id].setPos(*x.position)
			self.obstacles[x._id].setHpr(*x.rotaion)
			x._coll.setPos(*x.position)

	def raiseCollFlag(self,entry):
		self.collFlag=True
	def cheackCollision(self):
		if(self.collFlag):
			self.collFlag=False
			return True
		return False




	def renderFrame(self):
		#debug info
		self.disp.setText("dist: "+str(round(self.getGoalDist(),2))
		+"\npos: "+str(np.round(self.drone.position,3)))

		self.graphicsEngine.renderFrame()

	def getDepthBuffer(self,width=-1,hight=-1):
		#render scene witout goal
		#self.goalRender.detachNode()
		#self.renderFrame()
		#self.goalRender.reparentTo(self.render)

		data = self.depthTex.getRamImage()
		depth_image = np.frombuffer(data, np.float32)
		depth_image.shape = (self.depthTex.getYSize(), self.depthTex.getXSize(), self.depthTex.getNumComponents())
		depth_image = np.flipud(depth_image)
		#depth_image = depth_image / depth_image.max()

		#resize
		if width >0 and hight >0:
			tmpimg=Image.fromarray(depth_image.squeeze())
			tmpimg=tmpimg.resize((hight,width))
			depth_image=np.array(tmpimg)

		#rerender frame with goal
		#self.graphicsEngine.renderFrame()
		return depth_image

	def loadScene(self, scene : Scene, debug =False,showGoal=False):
		self.scene=scene
		for x in scene.walls:
			self.addWall(x,debug)

		for x in scene.obstacles:
			self.addObstacle(x,debug)

		self.drone.setPos(scene.startPos)
		self.drone.setRot(scene.startRot)

		self.addGoal(scene.goal,doesRender=showGoal)

	def unloadScene(self):
		self.scene=0

		for x in self.walls:
			x.removeNode()
		for x in self.obstacles:
			x.removeNode()
		for x in self.colliders:
			x.removeNode()

		self.walls=[]
		self.obstacles=[]
		self.obstaclesObj=[]
		self.drone.setPos(np.array([0,0,0]))
		self.drone.setRot(np.array([0,0,0]))
		self.goal=np.array([0,0,0])

	def getGoalDist(self):
		return (-self.drone.position[1]+self.goal[1])

	def getGoalDistFromStart(self):
		return (self.goal[1]-self.scene.startPos[1])

	def devAdd(self,entry):
		out=[]
		sanetised=""
		for char in entry:
			if(char.isnumeric() or char==","):
				sanetised+=char
		for x in sanetised.split(","):
			out.append(int(x))

		#wall=Wall(np.array(out))
		#wall.setPos(np.array([0,0,0]))
		self.devPos=np.array(out)

		self.setDevTools(False)
		self.devInputPos.enterText("Enter wall pos (speprated by commas):")
		self.devInputPos.show()



	def devSetPos(self,entry):
		out = []
		sanetised = ""
		for char in entry:
			if (char.isnumeric() or char == "," or char=="-" or char=="."):
				sanetised += char
		for x in sanetised.split(","):
			out.append(float(x))

		self.devInputPos.hide()

		wall=Wall(self.devPos)
		wall.setPos(np.array(out))
		print(wall.position)
		self.scene.addWall(wall)
		self.addWall(wall)


	def setDevTools(self,on):
		if(on):
			self.devInput.enterText('Enter wall dims (speprated by commas):')
			self.devInput.show()
		else:
			self.devInput.hide()

	def delLast(self):
		print(len(self.walls))
		if len(self.walls) >0:
			print("removed")
			self.walls[-1].removeNode()
			self.walls.pop(-1)
			self.colliders[-1].removeNode()
			self.colliders.pop(-1)
		if len(self.scene.walls) >0:
			self.scene.walls.pop(-1)
		
	

	
	
def simulate_network(engine):
		#inp=input("enter input:")
		is_down = engine.mouseWatcherNode.is_button_down

		if(is_down(KeyboardButton.ascii_key('w'))):
			return np.array([0,1,0,0,0,0])
		elif(is_down(KeyboardButton.ascii_key('s'))):
			return np.array([0,-1,0,0,0,0])
		elif(is_down(KeyboardButton.left())):
			return np.array([0, 0, 0, 10, 0, 0])
		elif (is_down(KeyboardButton.right())):
			return np.array([0, 0, 0, -10, 0, 0])
		elif (is_down(KeyboardButton.ascii_key('e'))):
			return np.array([0, 0, 0, 0, 0, 1])
		elif (is_down(KeyboardButton.ascii_key('q'))):
			return np.array([0, 0, 0, 0, 0, -1])
		elif(is_down(KeyboardButton.up())):
			return np.array([0, 0, 0, 0, 10, 0])
		elif (is_down(KeyboardButton.down())):
			return np.array([0, 0, 0, 0, -10, 0])
		elif (is_down(KeyboardButton.space())):
			return np.array([0, 0, 1, 0, 0, 0])
		elif (is_down(KeyboardButton.lshift())):
			return np.array([0, 0, -1, 0, 0, 0])
		elif(is_down(KeyboardButton.ascii_key("a"))):
			return np.array([-1, 0, 0, 0, 0, 0])
		elif (is_down(KeyboardButton.ascii_key("d"))):
			return np.array([1, 0, 0, 0, 0, 0])
		else:
			return np.array([0, 0, 0, 0, 0, 0])
		
def startMover(engine):
	is_down = engine.mouseWatcherNode.is_button_down
	return is_down(KeyboardButton.ascii_key("r"))

def devTools(engine):
	is_down = engine.mouseWatcherNode.is_button_down
	return is_down(KeyboardButton.ascii_key("f"))
def saveScene(engine):
	is_down = engine.mouseWatcherNode.is_button_down
	return is_down(KeyboardButton.ascii_key("v"))
def undo(engine):
	is_down = engine.mouseWatcherNode.is_button_down
	return is_down(KeyboardButton.ascii_key("z"))

















