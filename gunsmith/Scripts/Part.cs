using Godot;
using System;

public abstract partial class Part : Node3D
{
	public bool[][,] ShapeArrays;
	public Vector2I Index;
    [Export]
    public Vector3 TopLeftPosition { get; set; }//position of the center of the top-left most "cell" relative to the part origin, from origin to top-left
    [Export]
    public Vector3 BoundingBoxSize { get; set; }
    [Export]
    public Node3D Pivot { get; set; }
    private int ShapeRotation=0;
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
        Init();
    }
    public void Init() {
        ShapeArrays = new bool[4][,];
        init();
    }
    public virtual void init() {
        return;
    }
	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}
    public bool[,] GetShapeArray() {
        return ShapeArrays[ShapeRotation];
    }
    public void CreateShapeArrays(bool[,] BaseShape) {
        ShapeArrays[0] = BaseShape;
        for (int i = 1; i < 4; i++) {
            ShapeArrays[i] = RotateNonSquare(ShapeArrays[i - 1]);
        }
    }
    public bool[,] RotateNonSquare(bool[,] arr)
    {
        bool[,] TransposedArr = TransposeNonSquare(arr);
        int h = TransposedArr.GetLength(0);
        int w = TransposedArr.GetLength(1);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < (int)(w/2); j++) {
                bool tmp = TransposedArr[i, j];
                TransposedArr[i, j] = TransposedArr[i, w - j - 1];
                TransposedArr[i, w - j - 1] = tmp;
            }
        }
        return TransposedArr;
    }
    public bool[,] TransposeNonSquare(bool[,] arr)
    {
        int h = arr.GetLength(0);
        int w = arr.GetLength(1);
        bool[,] TransposedArr = new bool[w,h];//[h][w] -> [w][h]
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                TransposedArr[j, i] = arr[i, j];
            }
        }
        return TransposedArr;
    }
    public void rotateMatrixCCW(bool[,] mat)
    {
        int n = mat.GetLength(0);

        // Consider all cycles one by one
        for (int i = 0; i < n / 2; i++)
        {

            // Consider elements in group of 4
            // as P1, P2, P3 & P4 in current square
            for (int j = i; j < n - i - 1; j++)
            {

                bool temp = mat[i,j];
                mat[i,j] = mat[j,n - 1 - i];
                mat[j,n - 1 - i] = mat[n - 1 - i,n - 1 - j];
                mat[n - 1 - i,n - 1 - j] = mat[n - 1 - j,i];
                mat[n - 1 - j,i] = temp;
            }
        }
    }
    public void rotateMatrixCW(bool[,] mat)
    {
        int n = mat.GetLength(0);

        // Consider all cycles one by one
        for (int i = 0; i < n / 2; i++)
        {

            // Consider elements in group of 4
            // as P1, P2, P3 & P4 in current square
            for (int j = i; j < n - i - 1; j++)
            {

                bool temp = mat[i, j];
                mat[i, j] = mat[n - 1 - j, i];
                mat[n - 1 - j, i] = mat[n - 1 - i, n - 1 - j];
                mat[n - 1 - i, n - 1 - j] = mat[j, n - 1 - i];
                mat[j, n - 1 - i] = temp;
            }
        }
    }
    public void RotateShapeCW() {
        ShapeRotation = (ShapeRotation + 1) % 4;
    }
    public void RotateShapeCCW()
    {
        ShapeRotation--;
        if (ShapeRotation < 0) {
            ShapeRotation = 3;
        }
    }
    public void RotateCCW() {
        RotateShapeCCW();
        Pivot.RotateObjectLocal(Basis.Z, (float)(Math.PI/2));
        DebugPrint(ShapeArrays[ShapeRotation]);

    }
    public void RotateCW()
    {
        RotateShapeCW();
        Pivot.RotateObjectLocal(Basis.Z, -(float)(Math.PI / 2));
        DebugPrint(ShapeArrays[ShapeRotation]);
    }

    public Vector3 GetPartOrigin() {
        return TopLeftPosition.Rotated(new Vector3(0, 0, 1), (float.Pi / 2)*ShapeRotation).Abs(); // rotate TopLeftPosition to the current orientation and return the vector from the top-leftmost cell to the origin 
    }
    public void DebugPrint(bool[,] arr) {
        string printPayload = "";
        int h = arr.GetLength(0);
        int w = arr.GetLength(1);
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                printPayload += arr[i, j] + ",";
            }
            printPayload += "\n";
        }
        GD.Print(printPayload);
    }
}
