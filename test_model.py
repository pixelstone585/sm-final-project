
from simple_Network_sanitiy import *
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

LEVEL_LENGTH=1

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

SENSOR_DATA_SIZE=(600,600)

brain=drone_brain(explore_factor=0,explore_decay=0,explore_min=0,lr=0,input_size=SENSOR_DATA_SIZE,near_dist=engine.getNear(),far_dist=50)
best_reward=2

torch.autograd.set_detect_anomaly(True) #enable traceback for torch backprop errors

start_epoch=brain.load(r"D:\eyals_staff\final proj\Env V2\avgpool 23\model")



losses=[]

def gameLoop(engine,brain,epoch):
    engine.loadScene(scene1,debug=False)
    engine.drone.setPos(np.array([6,0,6]))
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
       
        tmp,raw_pred=brain.act(currState,False) #predict best move
        raw_pred=raw_pred.detach()
        tmp=brain.classDecodeExpiramntal2(np.argmax(raw_pred))
        #tmp=simulate_network(engine) #network interface here
        #print(raw_pred)
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
    brain.noLearn()
    engine.unloadScene()
    return generate_new
brain.explore_factor=0
epochs=100
#fancy progress bar 
best_reward=2
levels_beaten=0
tries=0
avg_tries=0
best_score_display=TextColumn("Best Reward: {task.fields[best_reward]}")
explore_factor_display=TextColumn("Explore Factor: {task.fields[explore_factor]}")
current_score_display=TextColumn("Current Reward: {task.fields[curr_reward]}")
levels_beaten_display=TextColumn("levels beaten: {task.fields[levels_beaten]}")
avg_tries_display=TextColumn("avg tries: {task.fields[avg_tries]}")

bar=Progress(TextColumn("[progress.description]{task.description}"),BarColumn(),MofNCompleteColumn(),best_score_display,explore_factor_display,current_score_display,levels_beaten_display,avg_tries_display)
bar_task=bar.add_task("running...",total=epochs,best_reward="N/A",explore_factor=brain.explore_factor,curr_reward="N/A",levels_beaten=0,avg_tries=0)
bar.start()
#training loop
for epoch in range(epochs):
    generate_new=gameLoop(engine,brain,epochs-epoch)
    best_reward=min(best_reward,min(brain.lifetime_rewards[-1]))
    bar.update(bar_task,advance=1,best_reward=round(best_reward,2),curr_reward=brain.lifetime_rewards[-1],explore_factor=brain.explore_factor,levels_beaten=levels_beaten,avg_tries=avg_tries)
    if generate_new:
        scene1,modules=compile_random_scene("level_modules",LEVEL_LENGTH)
        best_reward=0
        levels_beaten+=1
        avg_tries=(avg_tries*(levels_beaten-1)+tries)/levels_beaten#calculate new avarege tries
        tries=0
#plot rewards graph
rewards=[]
for x in brain.lifetime_rewards:
    rewards+=x
plt.plot(rewards)
plt.show()

plt.plot(losses)
plt.show()

#plot avarege reward per epoch
avgs=[]
for x in brain.lifetime_rewards:
    avgs.append(sum(x)/len(x))
plt.plot(avgs)
plt.show()

engine.destroy() #DO NOT REMOVE prevents spyder from entering a loop



