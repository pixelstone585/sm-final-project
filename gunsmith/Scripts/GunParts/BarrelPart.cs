using Godot;
using System;

public partial class BarrelPart : Part
{
	Connector Barrel1;
    Connector Barrel2;
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
        Barrel1 = GetNode<Connector>("Pivot/Mesh/BarrelConnector");
        Barrel2 = GetNode<Connector>("Pivot/Mesh/BarrelConnector2");
        Barrel1.PairConnector(Barrel2);
        Barrel1.PartUpdate += debug;
        Init();
    }
    public override void init()
    {
        bool[,] ShapeArray = {
            { true}
        };
        CreateShapeArrays(ShapeArray);
    }

    // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _Process(double delta)
	{
	}

    public void debug() {
        GetNode<MeshInstance3D>("Pivot/Mesh").Visible=false;
    }
}
