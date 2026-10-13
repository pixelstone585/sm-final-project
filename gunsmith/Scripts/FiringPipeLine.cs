using Godot;


public partial class FiringPipeLine : Node3D
{
    [Export]
    public float Pierce { get; set; } = 2f;
    [Export]
    public float Damage { get; set; } = 10f;
    [Export]
    public float MuzzleVelocity { get; set; } = 75;//in m/s
    [Export]
    public float RateOfFire { get; set; } = 100f;//in rpm
    public enum FireModes { Manual, semiAuto, FullAuto };
    [Export]
    public FireModes FireMode = FireModes.Manual;
    [Export]
    public float Range = 100f;
    [Export]
    public bool IsValid = false;

}
