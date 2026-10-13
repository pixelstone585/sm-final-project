using Godot;
using Godot.Collections;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

public partial class ItemIconGenerator : SubViewport
{
	public Godot.Collections.Dictionary<String, Texture2D> ItemIcons;
	public static ItemIconGenerator Instance;
	//I FUCKING HATE ASYNC FUNCTIONS
	//THAT INCLUDES YOU AWAIT.
	private Queue<PackedScene> RenderQueue;
	private bool isWorking = false;
	private Part PrevPart;
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		RenderQueue = new Queue<PackedScene>();
        ItemIcons = new Godot.Collections.Dictionary<String, Texture2D>();
		Instance = this;
		RenderTargetUpdateMode = UpdateMode.Once;
    }
	public async Task QueueIconGeneration(PackedScene PartScene) {
		isWorking = true;
		if (PrevPart != null) {
            PrevPart.QueueFree(); //clear previously renderd part
        }
		//PrevPart.Visible = false;
        Part part = PartScene.Instantiate<Part>();
		AddChild(part);
        RenderTargetUpdateMode = UpdateMode.Once;
		await ToSignal(RenderingServer.Singleton, RenderingServer.SignalName.FramePostDraw);
		Image tmp = GetTexture().GetImage();
        Texture2D IconTex = ImageTexture.CreateFromImage(tmp);//get texture from view port, dump image data and create new texture2d from raw data(create a copy)
        ItemIcons.Add(PartScene.ResourcePath, IconTex);
		PrevPart = part;
		isWorking = false;
	}
    public Texture2D GetIcon(PackedScene PartScene) {
		if (ItemIcons.ContainsKey(PartScene.ResourcePath)) {
            return ItemIcons[PartScene.ResourcePath];
        }
		RenderQueue.Enqueue(PartScene);
		return null;
	}
	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		
		if (isWorking == false)
		{
            PackedScene tmp;
			if (RenderQueue.TryDequeue(out tmp)) {
                QueueIconGeneration(tmp);
            }
			
		}
    }
}
