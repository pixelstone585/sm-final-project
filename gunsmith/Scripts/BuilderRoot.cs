using Godot;
using System;

public partial class BuilderRoot : Node3D
{
    [Signal]
    public delegate void ReturnPipelineEventHandler(FiringPipeLine Pipe);
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}
}
