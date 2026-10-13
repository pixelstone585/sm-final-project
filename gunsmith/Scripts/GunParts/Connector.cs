using Godot;
using System;
using Godot.Collections;
using System.Linq;

public partial class Connector : Area3D
{
	public FiringPipeLine PipeLine;
	[Export]
	public Array<Connector> PairedConnectors { get; set; }
	[Export]
	public ConnectionTypes Type { get; set; }
	[Signal]
	public delegate void PartUpdateEventHandler();

	public enum ConnectionTypes {LightAmmo,HeavyAmmo,ShotgunAmmo,Barrel};
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	public void PairConnector(Connector other) {
		if (!PairedConnectors.Contains(other)) {
            PairedConnectors.Add(other);
        }
		if (!other.PairedConnectors.Contains(this)) {
			other.PairedConnectors.Add(this);
		}
	}

	public void PropagatePipeLine(FiringPipeLine Pipe) {
		PropagatePipeLine(Pipe, new Array<Connector>());

    }

    //TODO: try to find an algorithem that does nor require cataloging previously treveresd Conectors
    public void PropagatePipeLine(FiringPipeLine Pipe, Array<Connector> TreveresdConnectors) {
        this.PipeLine = Pipe;
        //Signal that a PipeLine has been assigned
        EmitSignal(SignalName.PartUpdate);
		//add this node to the treversed nodes
		TreveresdConnectors.Add(this);
		//propagte to paired connectors
        foreach (Connector connector in PairedConnectors) {
			//propagate to connector if it is empty and has not been treversed
			if( !IsConnectorInGArray(connector, TreveresdConnectors))//&& 
            {
				connector.PropagatePipeLine(Pipe, TreveresdConnectors);
            }
		}
        Connector ExternalConnector = GetConnected();
		if (ExternalConnector !=null && !IsConnectorInGArray(ExternalConnector, TreveresdConnectors)) {
			ExternalConnector.PropagatePipeLine(Pipe, TreveresdConnectors);
        }

        
	}
	public void SetFiringPipeLine(FiringPipeLine Pipe) {
		this.PipeLine = Pipe;
        foreach (Connector connector in PairedConnectors)
        {
			//TODO: Figure out what to do when a Connector is Connected to 2 pipelines
			connector.PipeLine = Pipe;
        }
    }
	private bool IsConnectorInGArray(Connector Obj, Array<Connector> ObjArray) {
        foreach (Connector item in ObjArray)
        {
			if (item==Obj) {
				return true;
			}
        }
		return false;
    }

	public Connector GetConnected() {
		Array<Area3D> OverlappingAreas = GetOverlappingAreas();
		
		//search for overlapping connectors
		foreach (Area3D Area in OverlappingAreas) {
			if (Area is Connector) {
				Connector FoundConnector = (Connector)Area;
                //return if a valid connector was found and isnt in pairedConnectors
                if (!IsConnectorInGArray(FoundConnector,PairedConnectors)&& IsConnectorCompatible(FoundConnector))
				{
                    return FoundConnector;
                }
            }
		}
		return null;
	}

	public bool IsConnectorCompatible(Connector other) {
		bool ret = false;

		//check for full overlap
		//ret = ret || (other.GlobalPosition == GlobalPosition) && (other.Scale == Scale);
		ret = true;
		//check fo matching type
		ret = ret && (other.Type == Type);
		return ret;
	}

}
