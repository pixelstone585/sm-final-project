using Godot;
using System;

public partial class HitBox : Area3D
{
	[Signal]
	public delegate void DamageEventHandler(float Damage,float Pierce);
    [Export]
    public float PierceThreshold { get; set; }
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	public float Hit(float Damage,float Pierce) {
		GD.Print("Hit!");
		EmitSignal(SignalName.Damage, Damage,Pierce);
		return PierceThreshold;
	}
}
