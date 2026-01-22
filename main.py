
from simple_Network import *
from Environment import *
from State import State
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import loadPrcFileData
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, MofNCompleteColumn , TextColumn, BarColumn
print("sanity check")
loadPrcFileData("", "win-size 480 360")#set window size

drone=Drone()
engine=Engine(drone,debugCam=False,sensor_range=50)

scene =Scene()
testModule= Module()

scene.setStartPos(np.array([0,-5,0]))
#drone.setPos(np.array([0,-5,0]))
#wall=Wall(np.array([10,0,0]))
scene.setStartRot(np.array([0,0,0]))
#drone.setRot(np.array([0,0,0]))

wall=Wall(np.array([10,100,1]))
wall.setPos(np.array([-1,0,-2]))
scene.addWall(wall)

wall=Wall(np.array([10,100,1]))
wall.setPos(np.array([-1,0,9]))
scene.addWall(wall)

wall=Wall(np.array([1,100,10]))
wall.setPos(np.array([-2,0,-1]))
scene.addWall(wall)

wall=Wall(np.array([1,100,10]))
wall.setPos(np.array([9,0,-1]))
scene.addWall(wall)
#engine.addWall(wall,True)

wall=Wall(np.array([10,1,5]))
wall.setPos(np.array([-1,5,-1]))
scene.addWall(wall)

wall=Wall(np.array([5,1,10]))
wall.setPos(np.array([-1,10,-1]))
scene.addWall(wall)

obs=mover(np.array([1,1,1]))
obs.setPos(np.array([10,0,0]))
#scene.addObstacle(obs)

scene.setGoal(np.array([0,15,0]))
#engine.addObstacle(obs,True)
scene1=Scene()

LEVEL_LENGTH=2

scene1,modules=compile_random_scene("level_modules",LEVEL_LENGTH)



#drone.setPos(np.array([0,-15,0]))

#engine.addWall(wall)
#engine.addObstacle(obstacle)

#engine.registerWalls()
#engine.registerObstacles()



#engine.loadScene(scene1,debug=True)
#engine.unloadScene()

engine.renderFrame()


#engine.addObstacle(obs,showCollider=True)
#engine.setGoalRender(True) #<- rendering the goal is now off by defult for preformance reasons

#engine.addRuler(20)

SENSOR_DATA_SIZE=(30,40)
#orgenaisation, to be chaged for each run
RUN_IDETIFIER="basic CNN, 1"

NOTES="""
    basic network,\n
    explore factor 1, no decay
"""

#create files
os.mkdir(RUN_IDETIFIER)
f=open(RUN_IDETIFIER+r"/module_list","a")
dat=json.dumps(modules)
f.write(dat)
f.close()
f=open(RUN_IDETIFIER+r"/rewards.json","a")
f.write(json.dumps([]))
f.close()
f=open(RUN_IDETIFIER+r"/notes.txt","a")
f.write(NOTES)
f.close()

def save_rewards(path):
    rewards=[]
    for x in brain.lifetime_rewards:
        rewards+=x
    for x in brain.rewards:
        rewards+=x
    rewards=rewards.tolist()#no clue why rewards is a numpy array, but here is a fix regardless
    f=open(path,"w")
    f.write(json.dumps(rewards))
    f.close()

brain=drone_brain(explore_factor=1,explore_decay=0,explore_min=0,lr=0.1,input_size=SENSOR_DATA_SIZE)
best_reward=-2

torch.autograd.set_detect_anomaly(True) #enable traceback for torch backprop errors


losses=[]

def gameLoop(engine,brain,epoch):
    engine.loadScene(scene1,debug=False)
    engine.drone.setPos(np.array([4,0,4]))
    #engine.drone.setPos(np.array([6,3,6]))
    stop = False
    undoDelay=False
    generate_new=False
    while not stop:
        #update everything
        engine.tick()
        engine.updateCam()
        taskMgr.step()#panda3d's tick function
        engine.renderFrame()



        has_coll=engine.cheackCollision()
        if(has_coll):
            #print("crash!")
            stop=True

        if engine.getGoalDist() <=0:
            print("Success!")
            stop=True
            generate_new=True

        #generate state object from game data
        currState=State(engine.getDepthBuffer(*SENSOR_DATA_SIZE),has_coll,engine.getGoalDist() <=0,engine.getGoalDist(),engine.getGoalDistFromStart())
        reward=brain.analize_state(currState)
        #best_reward=max(best_reward,reward)
        #tmp=simulate_network(engine) #network interface here
        tmp=brain.act(currState) #predict best move
        tmp=tmp.squeeze()
        #dev tools
        #do NOT use while training!
        if(startMover(engine)): 
            #engine.addObstacle(obs,showCollider=True)
            plt.imshow(currState.sensorDat,cmap="Greys")
            plt.show()
        if(devTools(engine)):
            engine.setDevTools(True)

        if(saveScene(engine)):
            dat=engine.scene.saveAsModule()
            f=open(r"tmp.txt","w")
            f.write(json.dumps(dat))
            f.close()

        if(undo(engine)and not undoDelay):
            engine.delLast()
            undoDelay=True
        undoDelay=undo(engine)

        #preform movement
        engine.drone.move(np.clip(tmp[:3],-0.1,0.1))
        engine.drone.rotate(np.clip(tmp[3:],-0.5,0.5))
        engine.drone.rotation=np.clip(engine.drone.rotation,-50,50)

        bar.update(bar_task,advance=0,best_reward=round(best_reward,2),curr_reward=round(reward,2))

        #time.sleep(0.016666)
    #save stuff
    brain.save(RUN_IDETIFIER+r"/model",epoch)
    f=open(RUN_IDETIFIER+r"/module_list","w")
    dat=json.dumps(modules)
    f.write(dat)
    f.close()
    save_rewards(RUN_IDETIFIER+r"/rewards.json")
    losses.append(brain.learn())#backprop
    engine.unloadScene()
    return generate_new

epochs=50
#fancy progress bar 
best_reward=-2
best_score_display=TextColumn("Best Reward: {task.fields[best_reward]}")
explore_factor_display=TextColumn("Explore Factor: {task.fields[explore_factor]}")
current_score_display=TextColumn("Current Reward: {task.fields[curr_reward]}")

bar=Progress(TextColumn("[progress.description]{task.description}"),BarColumn(),MofNCompleteColumn(),best_score_display,explore_factor_display,current_score_display)
bar_task=bar.add_task("training...",total=epochs,best_reward="N/A",explore_factor=brain.explore_factor,curr_reward="N/A")
bar.start()
#training loop
for epoch in range(epochs):
    generate_new=gameLoop(engine,brain,epoch)
    best_reward=max(best_reward,max(brain.lifetime_rewards[-1]))
    bar.update(bar_task,advance=1,best_reward=round(best_reward,2),explore_factor=brain.explore_factor)
    if generate_new:
        scene1,modules=compile_random_scene("level_modules",LEVEL_LENGTH)
bar.stop()
#plot rewards graph
rewards=[]
for x in brain.lifetime_rewards:
    rewards+=x
plt.plot(rewards)
plt.show()

plt.savefig(RUN_IDETIFIER+"/reward_graph",format='svg')
plt.plot(losses)
plt.show()
plt.savefig(RUN_IDETIFIER+"/loss_graph",format='svg')

brain.save(RUN_IDETIFIER+r"/model")

engine.destroy() #DO NOT REMOVE prevents spyder from entering a loop



