using Godot;
using System;

public partial class LightChamber : Part
{
	Connector AmmoConnector;
	Connector BarrelConnector;

	
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		AmmoConnector = GetNode<Connector>("Pivot/PartMesh/AmmoConnector");
		BarrelConnector = GetNode<Connector>("Pivot/PartMesh/LBarrelConnector");
		Init();
    }
    public override void init() //called when Init() is called
    {
        bool[,] ShapeArray = new bool[,]
        {
            {true,true}
        };
        CreateShapeArrays(ShapeArray);
    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	
}
