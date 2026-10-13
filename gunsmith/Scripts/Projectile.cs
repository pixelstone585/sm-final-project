using Godot;
using Godot.Collections;
using System;
using System.Text.RegularExpressions;

public partial class Projectile : Node3D
{
	[Export]
	public float Speed { get; set; } = 10f;

	[Export]
	public float GravityAcc { get; set; } = 9.8f;
	[Export]
	public float Pierce { get; set; } = 2f;
	[Export]
	public float Damage { get; set; } = 10f;

	private float VisibleDelayCurrM;
	private float VisibleDelayM = 3f;
	private float CurrLifetimeSec;
	private float MaxLifeTimeSec;

	private float YSpeed = 0;
	private Vector3 LastPosition;
	private PhysicsDirectSpaceState3D StateSpace;
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		StateSpace = GetWorld3D().DirectSpaceState;

		MaxLifeTimeSec = 1000 / Speed;
        Visible = false;

		

    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

    public override void _PhysicsProcess(double delta)
    {
		YSpeed += (float)(GravityAcc * delta);
		LastPosition = GlobalPosition;
		GlobalPosition += GlobalBasis.Z * Speed*(float)delta + new Vector3(0, -YSpeed*(float)delta, 0);
		ResolveHits(LastPosition, GlobalPosition);
		
		//delay visibility
		Visible = true;

        //destroy projectile after speed/1km
        CurrLifetimeSec += (float)delta;
		if (MaxLifeTimeSec <= CurrLifetimeSec) {
			DestroyProjectile();
		}
    }

	public void ResolveHits(Vector3 From, Vector3 To) {
		Dictionary LastHit;
		Array<Rid> HitRids = new Array<Rid>();
		do {
			LastHit = RayCast(From, To, HitRids);
			if (LastHit.Count <= 0) {
				return;
			}
			Node HitNode = (Node3D)LastHit["collider"];
			if (!(HitNode is HitBox) && !(HitNode is Area3D)) {
				DestroyProjectile();
                return;
			}

            HitRids.Add( (Rid)LastHit["rid"]);
			if (HitNode is HitBox) {
                HitBox hitbox = HitNode as HitBox;
                Pierce -= hitbox.Hit(Damage, Pierce);
            }
			

			
		} while (Pierce>0);
		if (Pierce <= 0) {
			DestroyProjectile();
        }

    }
    public Dictionary RayCast(Vector3 From, Vector3 To,Array<Rid> Exclusions)
    {
        PhysicsRayQueryParameters3D query = PhysicsRayQueryParameters3D.Create(From, To,exclude: Exclusions);
		query.CollideWithAreas = true;
		return StateSpace.IntersectRay(query);
    }

	public void DestroyProjectile() {
		QueueFree();
	}
}
