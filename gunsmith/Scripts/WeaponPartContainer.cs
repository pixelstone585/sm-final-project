using Godot;
using System;

public partial class WeaponPartContainer : Node3D
{
	[Export]
	Connector MagConnector { get; set; }
    [Export]
    Connector BarrelConnector { get; set; }
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{

        MagConnector.PartUpdate += UpdatePipe;
        BarrelConnector.PartUpdate += UpdatePipe;
    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	public void UpdatePipe() {
		GD.Print(MagConnector.PipeLine + "," + BarrelConnector.PipeLine+"====");
		if (MagConnector.PipeLine != null && BarrelConnector.PipeLine != null) {
			MagConnector.PipeLine.IsValid = true;
		}
	}
}
