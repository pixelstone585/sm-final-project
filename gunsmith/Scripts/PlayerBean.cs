using Godot;
using System;

public partial class PlayerBean : CharacterBody3D
{
    [Export]
    public float WalkSpeed { get; set; } = 2;

    [Export]
    public float SprintSpeed { get; set; } = 4;

    [Export]
    public float SpeedAcc { get; set; } = 10;

    [Export]
    public int FallAcc { get; set; } = 9;

    [Export]
    public int JumpStrength { get; set; } = 10;

    [Export]
    public float MouseSensitivity { get; set; } = 1;

    [Export]
    public float CamLimit { get; set; } = 90f;

    [Export]
    public float BobStrength { get; set; } = 0.05f;

    [Export]
    public float BobFrequency { get; set; } = 3f;

    [Export]
    public float Fov { get; set; } = 75f;

    [Export]
    public float MaxFov { get; set; } = 150f;

    [Signal]
    public delegate void OpenWeaponBuilderEventHandler();

    private Vector3 _targetVelocity = Vector3.Zero;
    private Node3D Pivot;
    private Camera3D Camera;
    private Weapon PWeapon;

    private float BobAcumulator = 0;

    private bool IsBuilderOpen = false;


    public override void _Ready()
    {
        Pivot = GetNode<Node3D>("Pivot");//get pivot node
        Camera = Pivot.GetNode<Camera3D>("PlayerCam");//get camera node
        PWeapon = GetNode<Weapon>("Pivot/Weapon");
        Input.MouseMode = Input.MouseModeEnum.Captured;//lock mouse
    }

    public override void _Process(double delta)
    {
        if (Input.IsActionJustPressed("ui_cancel")) {
            Input.MouseMode = Input.MouseModeEnum.Visible;//release mouse
        }
    }

    //update physics
    public override void _PhysicsProcess(double delta)
    {
        if (Input.IsActionJustPressed("open_builder")) {
            ToggleBuilder();
        }
        if (!IsBuilderOpen) {
            walk(delta);
            cameraBobing(delta);
        }
       
    }


    public override void _Input(InputEvent @event)//note to self: @name is used when name is a reserved keyword
    {
        if (@event is InputEventMouseMotion && !IsBuilderOpen) {
            look(@event as InputEventMouseMotion);//pass @event as type InputEventMouseMotion
        }
    }

    public void look(InputEventMouseMotion @event) {
        Vector2 DMousePos = @event.Relative;

        RotateY(Mathf.DegToRad(-DMousePos.X * MouseSensitivity));//left-right rotation

        float UDChange = -DMousePos.Y * MouseSensitivity;
        if (Pivot.RotationDegrees.X + UDChange < CamLimit && Pivot.RotationDegrees.X + UDChange > -CamLimit) {
            Pivot.RotateX(Mathf.DegToRad(UDChange));
        }
    }

    //handles walking and jumping
    public void walk(double delta)
    {
        Vector3 direction = Vector3.Zero;
        //read input
        if (Input.IsActionPressed("move_left"))
        {
            direction -= Basis.X;
        }

        if (Input.IsActionPressed("move_right"))
        {
            direction += Basis.X;
        }

        if (Input.IsActionPressed("move_forward"))
        {
            direction -= Basis.Z;
        }

        if (Input.IsActionPressed("move_backward"))
        {
            direction += Basis.Z;
        }

        //normalize direction vector
        if (direction != Vector3.Zero)
        {
            direction = direction.Normalized();
        }

        //compute target velocity
        if (Input.IsActionPressed("sprint"))
        {
            _targetVelocity.X = direction.X * SprintSpeed;
            _targetVelocity.Z = direction.Z * SprintSpeed;
        }
        else {
            _targetVelocity.X = direction.X * WalkSpeed;
            _targetVelocity.Z = direction.Z * WalkSpeed;
        }


        if (!IsOnFloor())
        {
            _targetVelocity.Y -= FallAcc * (float)delta;
        }

        if (IsOnFloor() && Input.IsActionPressed("jump")) {
            _targetVelocity.Y = JumpStrength;
        }

        //apply movement
        Velocity = _targetVelocity;//CharacterBody3D.Velocity
        MoveAndSlide(); //let the engine do the hard work

    }

    public void cameraBobing(double delta) {
        if (!IsOnFloor()) { return; }//dont bob in air
        //ajust fov based on speed
        float NewFov = Fov * ((Velocity.Length() - WalkSpeed)/1.81818181818f);
        NewFov=Math.Clamp(NewFov, Fov, MaxFov);
        Camera.Fov = Mathf.Lerp(Camera.Fov, NewFov, (float)delta*5);

       
        BobAcumulator += (float)delta * Velocity.Length();
        BobAcumulator = BobAcumulator % (2 * (float)Math.PI);//reset acumulator every 360 cycle to avoid float imprecition

        float SineY = (float)Math.Sin(BobAcumulator*BobFrequency) *BobStrength;
        float CosX = (float)Math.Sin(BobAcumulator * BobFrequency / 2) * BobStrength;
        Camera.Position = new Vector3(CosX, SineY, Camera.Position.Z);


    }

    public void ToggleBuilder() {
        if (!IsBuilderOpen)
        {
            Input.MouseMode = Input.MouseModeEnum.Visible;//release mouse
            PWeapon.LoadBuilder();
            PWeapon.builderroot.GetNode<Camera3D>("BuilderCam").MakeCurrent();
            IsBuilderOpen = true;
            PWeapon.AllowFiring(false);
        }
        else {
            Input.MouseMode = Input.MouseModeEnum.Captured;//release mouse
            Camera.MakeCurrent();
            PWeapon.UnLoadBuilder();
            IsBuilderOpen = false;
            PWeapon.AllowFiring(true);
        }
    }
}
