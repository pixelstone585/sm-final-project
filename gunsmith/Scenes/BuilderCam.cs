using Godot;
using Godot.Collections;
using System;
using System.Runtime.InteropServices;

public partial class BuilderCam : Camera3D
{
	PhysicsDirectSpaceState3D SpaceState;
	Vector2 MousePos;
    // Called when the node enters the scene tree for the first time.
    public override void _Ready()
	{
		SpaceState = GetWorld3D().DirectSpaceState;
		MousePos = GetViewport().GetMousePosition();
    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

	public Dictionary MouseRayCast(float Length, bool CollideWithAreas,uint bitmask) {
        MousePos = GetViewport().GetMousePosition();//update mouse pos
		Vector3 Origin = ProjectRayOrigin(MousePos);//find ray origin in 3d space
		Vector3 End = Origin + ProjectRayNormal(MousePos) * Length;//calculate end position

		PhysicsRayQueryParameters3D query = PhysicsRayQueryParameters3D.Create(Origin, End);//create raycat query
        query.CollisionMask = bitmask;
		query.CollideWithAreas = CollideWithAreas;//enable collision with areas

		return SpaceState.IntersectRay(query);//calculate and return result
    }
	public Dictionary MouseRayCast(float Length, bool CollideWithAreas) {
        MousePos = GetViewport().GetMousePosition();//update mouse pos
        Vector3 Origin = ProjectRayOrigin(MousePos);//find ray origin in 3d space
        Vector3 End = Origin + ProjectRayNormal(MousePos) * Length;//calculate end position

        PhysicsRayQueryParameters3D query = PhysicsRayQueryParameters3D.Create(Origin, End);//create raycat query
        query.CollideWithAreas = CollideWithAreas;//enable collision with areas

        return SpaceState.IntersectRay(query);//calculate and return result
    }
    public Vector3? MouseRayCastIntersectPlane(Plane plane)
    {
        MousePos = GetViewport().GetMousePosition();//update mouse pos
        Vector3 Origin = ProjectRayOrigin(MousePos);//find ray origin in 3d space
        Vector3 Direction =  ProjectRayNormal(MousePos);//calculate end position


        return plane.IntersectsRay(Origin,Direction);//calculate and return result
    }
}
