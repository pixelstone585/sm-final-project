using Godot;
using Godot.Collections;
using System;

public partial class BuildInitiator : Node3D
{
	[Export]
	Array<Connector> Connectors { get; set; }

    public BuilderRoot ReturnSignalEmitter;
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	public void Build() {
        FiringPipeLine Pipe = new FiringPipeLine();
        foreach (Connector conn in Connectors)
        {
            conn.PropagatePipeLine(Pipe);
        }
        GD.Print(Pipe.IsValid);
        ReturnSignalEmitter.EmitSignal(BuilderRoot.SignalName.ReturnPipeline, Pipe);
    }

    public void ConnectBuildButton(Button button)
    {
        if (!button.IsConnected(Button.SignalName.Pressed, Callable.From(Build))) {
            button.Pressed += Build;
        }
        GD.Print("Connected!");
    }
}
