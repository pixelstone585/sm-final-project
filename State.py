#fancy container for info regarding the current state
class State():
    def __init__(self,sensorDat,has_crashed,is_sucesssful,goalDist,goalDistFromStart):
        self.sensorDat=sensorDat
        self.has_crashed=has_crashed
        self.is_sucesssful=is_sucesssful
        self.goalDist=goalDist
        self.goalDistFromStart=goalDistFromStart