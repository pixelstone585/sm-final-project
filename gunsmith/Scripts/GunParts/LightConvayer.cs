using Godot;
using System;

public partial class LightConvayer : Part
{
	Connector Ammo1;
	Connector Ammo2;
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		Ammo1 = GetNode<Connector>("Pivot/Mesh/AmmoConnector");
        Ammo2 = GetNode<Connector>("Pivot/Mesh/AmmoConnector2");
		Ammo1.PairConnector(Ammo2);
        Init();
	}
	public override void init() {
        bool[,] ShapeArray = {
            { true,true}
        };
        CreateShapeArrays(ShapeArray);
    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}
}
