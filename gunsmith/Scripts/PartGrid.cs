using Godot;
using Godot.Collections;
using System;
using System.Reflection;
using System.Xml;

public partial class PartGrid : Sprite3D
{
	[Export]
	Vector2I GridSize { get; set; }
	[Export]
	BuilderCam Camera { get; set; }
	[Export]
	Button BuildButton { get; set; }
	[Export]
	BuilderRoot BuilderRootNode { get; set; }

	Vector2 SLOTSIZEPIX;
	Part[,] PartArray;

	Area3D GridArea;
	CollisionShape3D GridCollisionShape;

	Part HeldPart;
	Vector3 GrabPoint;
	Vector3 GrabPlaceOffset;

	

	private uint RayCastMask;
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		//get nodes
		GridArea = GetNode<Area3D>("GridArea");
		GridCollisionShape = GridArea.GetNode<CollisionShape3D>("GridCollisionShape");
        //create part array
        PartArray = new Part[GridSize.Y, GridSize.X];
		//set collision mask(layer 16)
		RayCastMask = 0b00000000_00000000_10000000_00000000;
		//--------------------------------
		//configure gird cell size
		SLOTSIZEPIX = new Vector2(Texture.GetWidth(), Texture.GetHeight());
		RegionRect = new Rect2(Vector2.Zero, SLOTSIZEPIX * GridSize);
		PixelSize = 1 / SLOTSIZEPIX[0];

		//configure area3d
		GridArea.Position = new Vector3(GridSize[0] / 2, GridSize[1] / 2, 0);//position area in the middle of the grid
		//configure area3d shape
		BoxShape3D areaShape = new BoxShape3D();
		areaShape.Size = new Vector3(GridSize[0], GridSize[1],0.1f);
		GridCollisionShape.Shape =areaShape;
    }

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}

    public override void _PhysicsProcess(double delta)
    {
		
		//drag and drop parts
		if (Input.IsActionJustPressed("pick_up_part")) {
            HeldPart=PickUpPart();
		}
		if (Input.IsActionPressed("pick_up_part")&&HeldPart!=null) {
            Dictionary RayCastResult = Camera.MouseRayCast(20, true,RayCastMask);
			if (RayCastResult.Count > 0)
			{
				Vector3 PartPosition = (Vector3)RayCastResult["position"];
				HeldPart.Position = ToLocal(PartPosition)-GrabPoint;
			}
        }
		if (Input.IsActionJustReleased("pick_up_part") && HeldPart != null ) { 
			Vector2I PlaceIndex = RayCastFindIndex(GrabPlaceOffset);
			
			if (!(PlaceIndex[0] == -1 || PlaceIndex[1] == -1) && CanPlace(HeldPart, PlaceIndex))
			{
				GD.Print(PlaceIndex);
				PlacePart(HeldPart, PlaceIndex);
            }
			else {
                ReturnPartToInventory(HeldPart);
            }
		}

		//rotate held part
		if (Input.IsActionPressed("pick_up_part") && HeldPart != null && Input.IsActionJustPressed("rotate_part_counter_clockwise")) {
            HeldPart.RotateCCW();

		}
        if (Input.IsActionPressed("pick_up_part") && HeldPart != null && Input.IsActionJustPressed("rotate_part_clockwise"))
        {
            HeldPart.RotateCW();
        }
    }
	//REMEMBER!!!! X=WIDTH=INDEX1 Y=HEIGHT=INDEX0
	public void PlacePart(Part part, Vector2I index)
	{
		part.Index = index;
		bool[,] PartShape = part.GetShapeArray();
		int LenY = PartShape.GetLength(0);
        int LenX = PartShape.GetLength(1);
		int ArrLenY = PartArray.GetLength(0);
        int ArrLenX = PartArray.GetLength(1);
		//failsafe: escape if placing would require going out of bounds
		if (index.X + LenX - 1 >= ArrLenX || index[1] + LenY - 1 >= ArrLenY) {
			ReturnPartToInventory(part);
            return;
		}

        for (int iy = 0; iy < LenY; iy++){
			for (int ix = 0; ix < LenX; ix++) {
				if (PartShape[iy, ix]) {
					PartArray[index.Y + iy, index.X + ix] = part;
				}
			}
		}
		Vector3 PartOriginLocal = part.GetPartOrigin();
		part.Position = new Vector3(index[0]+ PartOriginLocal.X+0.5f,  index[1] + PartOriginLocal.Y+0.5f,0.6f);

		if (part.IsInGroup("Chamber")) {
			BuildInitiator bi = (BuildInitiator)part.FindChild("BuildInitiator");
			bi.ReturnSignalEmitter = BuilderRootNode;
			bi.ConnectBuildButton(BuildButton);
		}
    }

	public bool CanPlace(Part part, Vector2I index) {
		if (index[0] == -1 && index[1] == -1) { return false; }

		bool[,] PartShape = part.GetShapeArray();
        int LenY = PartShape.GetLength(0);
        int LenX = PartShape.GetLength(1);
        int ArrLenY = PartArray.GetLength(0);
        int ArrLenX = PartArray.GetLength(1);
        //return false if placing would require going out of bounds
        if (index.X + LenX - 1 >= ArrLenX || index[1] + LenY - 1 >= ArrLenY)
        {
			GD.Print(index+","+(index.X + LenX - 1)+","+ ArrLenX);
            return false;
        }

        for (int iy = 0; iy < LenY; iy++)
        {
            for (int ix = 0; ix < LenX; ix++)
            {
                if (PartShape[iy, ix])
                {
                    if (PartArray[index.Y + iy, index.X + ix] != null)
                    {
                        return false;
                    }
                }
            }
        }
		return true;
    }

	public Part RemovePart(Vector2I Index) {
		Part part = PartArray[Index.Y, Index.X];
		if (part == null) {
			return null;
		}
		Vector2I PartIndex = part.Index;
        bool[,] PartShape = part.GetShapeArray();
        int LenY = PartShape.GetLength(0);
        int LenX = PartShape.GetLength(1);
        for (int iy = 0; iy < LenY; iy++)
        {
            for (int ix = 0; ix < LenX; ix++)
            {
                if (PartShape[iy, ix])
                {
                    PartArray[PartIndex.Y + iy, PartIndex.X + ix] = null;
                }
            }
        }
		return part;
    }

	public void ReturnPartToInventory(Part part) {
		part.QueueFree();
		//ADD INVENTORY CODE HERE
	}

	public Vector2I GetIndexFromPosition(Vector2 Position) {
		Vector2I index = (Vector2I)Position;

        if (index[0] < GridSize[0] && index[0] >=0 && index[1] < GridSize[1] && index[1] >=0 ) {//cheack that the result is in array bounds
			return index;
		}
		return -Vector2I.One;//return (-1,-1) if position is out of bounds
	}
	public Vector2I GetIndexFromPosition(Vector3 WorldPosition) {
		Vector3 localPos = ToLocal(WorldPosition);//convert to local coordinants
		GD.Print(localPos+"--");
		return GetIndexFromPosition(new Vector2(localPos[0], localPos[1]));//get xy componants
    }

    public Part PickUpPart() {
		//handle taking parts from inventory
		Control UiClickedOn = GetViewport().GuiGetHoveredControl();
        if ( UiClickedOn != null && UiClickedOn.Owner is ItemUi) {
			Part part = CreatePart(((ItemUi)UiClickedOn.Owner).PartScene);

			GrabPlaceOffset = Vector3.Zero; 
            return part;
        }
		//handle taking item from grid
        Dictionary RayCastResult = Camera.MouseRayCast(20, true,RayCastMask);
        if (RayCastResult.Count > 0 && GridArea == (Node3D)RayCastResult["collider"])
        {
			GrabPoint = (Vector3)RayCastResult["position"];
            Vector2I index = GetIndexFromPosition(GrabPoint);
			Part part= RemovePart(index);
			if (part == null) {
				return null;
			}

			//figure out where the part is bieng grabbed from
            GrabPoint = ToLocal(GrabPoint)-part.Position ;
            GrabPoint.Z = 0;
			GrabPoint = GrabPoint.Snapped(0.5f);//snap to 0.5 incraments
			Vector3 OriginLocal = part.GetPartOrigin();
			GrabPlaceOffset = OriginLocal + GrabPoint;
            return part;
        }
		return null;
    }

	public Part CreatePart(PackedScene PartScene) {
		Part NewPart = PartScene.Instantiate<Part>();
		SetRenderingLayerRecursive(NewPart);
		AddChild(NewPart);
		return NewPart;
	}

	public Vector2I RayCastFindIndex(Vector3 Offset) {
        Dictionary RayCastResult = Camera.MouseRayCast(20, true,RayCastMask);
        if (RayCastResult.Count > 0 && GridArea == (Node3D)RayCastResult["collider"])
        {
			Vector3 RayCastResultRaw = (Vector3)RayCastResult["position"];
			Vector3 RayCastResultOffset = new Vector3(RayCastResultRaw.X + Offset.X, RayCastResultRaw.Y, RayCastResultRaw.Z + Offset.Y);
            Vector2I index = GetIndexFromPosition(RayCastResultOffset);
			GD.Print(index+","+ Offset+","+ RayCastResultOffset);
            return index;
        }
		return -Vector2I.One;
    }

	public void SetRenderingLayerRecursive(Node node) {
		if (node == null) { return; }
		if (node is VisualInstance3D) {
			VisualInstance3D Vnode = (VisualInstance3D)node;
			Vnode.SetLayerMaskValue(2, true);
		}

		if (node.GetChildCount() > 0) { 
			Array<Node> Children = node.GetChildren();
			foreach (Node item in Children)
			{
				SetRenderingLayerRecursive(item);
			}
		}
	}
}
