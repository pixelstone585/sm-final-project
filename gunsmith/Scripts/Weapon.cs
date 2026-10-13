using Godot;
using System;
using System.Runtime.CompilerServices;
using static FiringPipeLine;
using static System.Net.Mime.MediaTypeNames;

public partial class Weapon : Node3D
{
    [Export]
    public Vector3 MuzzlePos { get; set; }
    [Export]
    public Node3D ProjectileSpawnerNode { get; set; }

    public bool CanFire = true;

    private FiringPipeLine PipeLine;

    public GpuParticles3D MuzzleFlashParticle;

    private double FireDelay;
    private double FireTimer=0;
    private PackedScene ProjectileScene;

    public BuilderRoot builderroot { get; private set; }

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
        //debugging
        FiringPipeLine tmp = GetNode<FiringPipeLine>("TestingPipeLine");
        RegisterPipeLine(tmp);

        //get muzzle flash particle
        MuzzleFlashParticle = GD.Load<PackedScene>("res://Scenes/muzzle_flash.tscn").Instantiate<GpuParticles3D>();
        MuzzleFlashParticle.OneShot = true;//set it so the emitter shots out all particles at once.
        AddChild(MuzzleFlashParticle);
        MuzzleFlashParticle.Position = MuzzlePos;

        //create builder instance
        builderroot = GD.Load<PackedScene>("res://Scenes/builder_root.tscn").Instantiate<BuilderRoot>();

        //load Projectile Scene
        ProjectileScene = GD.Load<PackedScene>("res://Scenes/projectile.tscn");
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

    public override void _PhysicsProcess(double delta)
    {
        switch (PipeLine.FireMode)
        {

            case FireModes.Manual:
                SemiAuto(delta);
                //add chambering animation here
                break;

            case FireModes.semiAuto:
                SemiAuto(delta);
                break;

            case FireModes.FullAuto:
                FullAuto(delta);
                break;

            default:
                GD.PushWarning("weapon fire mode is not manual,semi-auto,or full auto. how did you manage that?");
                break;
        }
    }

    public void RegisterPipeLine(FiringPipeLine Pipe) {
        PipeLine = Pipe;
        AllowFiring(CanFire);
        //setup variables
        FireDelay = 1 / (60 * Pipe.RateOfFire);//get the delay between bullets from the rate of fire(rpm -> s/round)
    }
    public void ClearPipeLines() {
        PipeLine = null;
    }

    private void Fire()
    {
        //GD.Print("BANG!");
        Projectile NewProjectile = ProjectileScene.Instantiate<Projectile>();
        GetTree().CurrentScene.AddChild(NewProjectile);//reparents to root scene

        NewProjectile.GlobalPosition = ProjectileSpawnerNode.GlobalPosition;
        NewProjectile.LookAt(ProjectileSpawnerNode.ToGlobal(-Vector3.Forward));

        NewProjectile.Damage = PipeLine.Damage;
        NewProjectile.Pierce = PipeLine.Pierce;
        NewProjectile.Speed = PipeLine.MuzzleVelocity;
    }

    public void SemiAuto(double delta) {
        //reset muzzle flash one tick after it was used
        if (FireTimer == FireDelay) {
            MuzzleFlashReset();
        }
        if (FireTimer > 0) { FireTimer -= delta; }//step timer
        //fire if input was presssed and if the delay has passed
        if (Input.IsActionJustPressed("fire_weapon") && FireTimer <= 0 && CanFire) {
            Fire();
            FireTimer = FireDelay;
            MuzzleFlash();
        }
    }

    public void FullAuto(double delta) {
        //reset muzzle flash one tick after it was used
        if (FireTimer == FireDelay)
        {
            MuzzleFlashReset();
        }
        if (FireTimer > 0) { FireTimer -= delta; }//step timer
        //fire if input was presssed and if the delay has passed
        if (Input.IsActionPressed("fire_weapon") && FireTimer <= 0 && CanFire)
        {
            Fire();
            FireTimer = FireDelay;
            MuzzleFlash();
        }
    }
    private void MuzzleFlashReset()
    {
        MuzzleFlashParticle.Emitting = false;
        MuzzleFlashParticle.Restart();
    }
    private void MuzzleFlash()
    {
        MuzzleFlashParticle.Emitting = true;
    }

    public void LoadBuilder() {
        AddChild(builderroot);
        builderroot.ProcessMode = ProcessModeEnum.Inherit;//unpause processing
        //connect signal
        builderroot.ReturnPipeline += SetPipeLine;
    }

    public void UnLoadBuilder() {
        RemoveChild(builderroot);
        builderroot.ProcessMode = ProcessModeEnum.Disabled;//pause processing
        GD.Print("unload");
    }

    public void SetPipeLine(FiringPipeLine Pipe) {
        RegisterPipeLine(Pipe);
    }

    public void AllowFiring(bool Allow) {
        CanFire = Allow && PipeLine.IsValid;
    }

}
