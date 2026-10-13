using Godot;
using System;

public partial class HpTracker : Node3D
{
    [Export]
    public float Health { get; set; } = 100;
    [Export]
    public float Armor { get; set; } = 1f;
    [Export]
    public float PierceThreshold { get; set; } = 1.1f;
    [Export]
    public int IMSecs { get; set; } = 100;
    [Export]
    public String Labeltext { get; set; } = "Hp: ";
    [Export]
    public Label3D Label { get; set; }
    [Export]
    public HitBox LinkedHitBox { get; set; }

    public float CurrHealth;

    private ulong LastHitTimeMSec = 0;

    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
        CurrHealth = Health;
        UpdateLinkedLabel3D();
        if (LinkedHitBox != null) {
            LinkHitBox(LinkedHitBox);
        }
    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

    public void Damage(float Damage,float Pierce) {
        if ((int)(Time.GetTicksMsec() - LastHitTimeMSec) >= IMSecs) {
            LastHitTimeMSec = Time.GetTicksMsec();
            CurrHealth -= Damage*Math.Min(1,Pierce/Armor);//reduce damage taken by pierce relative to armor
        }
        UpdateLinkedLabel3D();
    }

    public void UpdateLinkedLabel3D() {
        if (Label != null) {
            Label.Text = Labeltext + CurrHealth;
        }
    }

    public void LinkHitBox(HitBox hitbox,bool OverridePierceThreshold=true) {
        if (OverridePierceThreshold) {
            hitbox.PierceThreshold = PierceThreshold;
        }
        hitbox.Damage += Damage;
    }
}
