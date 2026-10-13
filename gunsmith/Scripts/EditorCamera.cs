using Godot;
using System;

public partial class EditorCamera : Node3D
{
    [Export]
    public float MouseSensitivity { get; set; }=1;
    [Export]
    public float CamYLimit { get; set; } = 90f;
    [Export]
    public float CamMaxIn { get; set; } = 1;
    [Export]
    public float CamMaxOut { get; set; } = 10;
    private Node3D PivotY;
    private Camera3D Camera;
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
		Input.MouseMode = Input.MouseModeEnum.Visible;//release mouse
        PivotY = GetNode<Node3D>("CameraPivotY");
        Camera = GetNode<Camera3D>("CameraPivotY/Camera");
    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}
    public override void _PhysicsProcess(double delta)
    {
        if (Input.IsActionJustReleased("builder_cam_out") && Camera.Position.Z+0.2 <= CamMaxOut) {
            Camera.Position += new Vector3(0f, 0f, 0.2f);
        }
        if (Input.IsActionJustReleased("builder_cam_in") && Camera.Position.Z + 0.2 >= CamMaxIn)
        {
            Camera.Position += new Vector3(0f, 0f, -0.2f);
        }
        if (Input.IsActionJustPressed("builder_cam_orbit")) {
            Input.MouseMode = Input.MouseModeEnum.Captured;//lock mouse
        }
        if (Input.IsActionJustReleased("builder_cam_orbit")) {
            Input.MouseMode = Input.MouseModeEnum.Visible;//release mouse
        }
    }

    public override void _Input(InputEvent @event)//note to self: @name is used when name is a reserved keyword
    {
        if (@event is InputEventMouseMotion && Input.IsActionPressed("builder_cam_orbit"))
        {
            look(@event as InputEventMouseMotion);//pass @event as type InputEventMouseMotion
        }
    }

    public void look(InputEventMouseMotion @event) {
        Vector2 DMousePos = @event.Relative;

        RotateY(Mathf.DegToRad(-DMousePos.X * MouseSensitivity));//left-right rotation
        float UDChange = -DMousePos.Y * MouseSensitivity;
        if (PivotY.RotationDegrees.X + UDChange < CamYLimit && PivotY.RotationDegrees.X + UDChange > -CamYLimit)
        {
            PivotY.RotateX(Mathf.DegToRad(UDChange));
        }
    }
}
