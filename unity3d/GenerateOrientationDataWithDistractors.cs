using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using System.IO;
public class GenerateOrientationDataWithDistractors : MonoBehaviour
{
    public string BackGroundImagesPath = "/home/viktor/Projects/Unity3D/ml-imagesynthesis/Assets/Resources/BackgroundImages";
    // /home/goat/Projects/Unity/ml-imagesynthesis/Assets/Resources/BackgroundImages
    public bool HasDistractorObjects = true;
    public bool HasDistractorVehicles = true;
    private GameObject[] prefabs;
    private GameObject[] prefabsDistractors;
    private GameObject Distractor;
    private GameObject[] DistractorVehicles = new GameObject[3];

    // Start is called before the first frame update
    private GameObject newObj;
    private int prefabIndex = 0;
    private Vector3 newPos;
    private Quaternion newRot;
    private Vector3 newPosDistractor;
    private Quaternion newRotDistractor;
    private Vector3 newPosDistractorVehicle;
    private Quaternion newRotDistractorVehicle;
    private float scaleFactor;
    private GameObject[] cubes = new GameObject[8];
    private int countImages = 0;
    private Renderer rend;

    private int BgIndex = 0;
    public int resWidth = 1920;
    public int resHeight = 1080;
    private float bboxMinX;
    private float bboxMinY;
    private float bboxMaxX;
    private float bboxMaxY;
    List<string> ImagePaths;
    public int resWidthBbox;
    public int resHeightBbox;
    private Camera camerax;
    private GameObject lightGameObject;
    private Light Lightx;
    public Shader uberReplacementShader;
    private GameObject backgroundQuat;
    private float alpha;
    private float beta;
    private float beta1;
    private float beta2;
    private int indexDistractor;
    private int indexDistractorVehicle;
    private float alphaDistractor;

    private void SetBackgroundImagePaths()
    {
        ImagePaths = new List<string>();
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi1.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi2.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi3.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi4.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi5.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi6.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi7.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi8.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi9.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi10.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi11.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi12.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi13.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi14.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi15.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi16.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi17.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi18.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi19.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi20.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi21.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi22.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi23.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi24.jpg")); ;
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi25.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi26.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi27.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi28.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi29.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi30.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi31.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi32.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi33.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi34.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi35.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi36.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi37.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi38.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi39.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi40.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi41.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi42.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi43.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi44.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi45.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi46.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi47.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi48.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi49.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi50.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi51.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi52.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi53.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi54.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi55.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi56.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi57.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi58.jpg"));
        ImagePaths.Add(Path.Combine(BackGroundImagesPath, "avi59.jpg"));
    }

    void Start()
    {
        // Make a game object
        lightGameObject = new GameObject("The Light");
        // Add the light component
        Lightx = lightGameObject.AddComponent<Light>();
        Lightx.type = LightType.Directional;
        Lightx.color = new Color(1f, 1f, 1f);
        Lightx.intensity = 1.21f;
        Lightx.transform.position = new Vector3(0f, 0f, -12f);
        Lightx.transform.eulerAngles = new Vector3(0f, 0f, 0f);
        camerax = GetComponent<Camera>();

        // prefabs = Resources.LoadAll<GameObject>("CubesV3");
        // prefabs = Resources.LoadAll<GameObject>("CubesV1");
        // prefabs = Resources.LoadAll<GameObject>("SphereV");
        prefabs = Resources.LoadAll<GameObject>("Cars_folder");
        prefabsDistractors = Resources.LoadAll<GameObject>("Distractors/- Prefabs/Props");
        Destroy(newObj);

        SetBackgroundImagePaths();
    }

    private void LoadImageWithIndex(int Index)
    {
        // var filePath = "/home/viktor/Projects/VehicleX/VehicleX/Background_imgs/vdo (1).avi/avi(2).jpg";
        string filePath = ImagePaths[Index];
        if (System.IO.File.Exists(filePath))
        {
            var bytes = System.IO.File.ReadAllBytes(filePath);
            var tex = new Texture2D(1, 1);
            tex.LoadImage(bytes);
            Destroy(backgroundQuat);
            backgroundQuat = GameObject.CreatePrimitive(PrimitiveType.Quad);
            backgroundQuat.transform.position = new Vector3(0, 0, 27.4f);
            backgroundQuat.transform.localScale = new Vector3(76.8f, 43.2f, 1);
            Collider backgroundQuatCollider = backgroundQuat.GetComponent<Collider>();
            Destroy(backgroundQuatCollider);
            MeshRenderer backgroundQuatMeshRenderer = backgroundQuat.GetComponent<MeshRenderer>();
            backgroundQuatMeshRenderer.receiveShadows = false;
            backgroundQuatMeshRenderer.materials[0].mainTexture = tex;
            // backgroundQuatMeshRenderer.materials[0].EnableKeyword("_EMISSION");
            // backgroundQuatMeshRenderer.materials[0].SetColor("_EmissionColor", new Color(0.6f,0.6f,0.6f));// das macht das bild heller
            backgroundQuatMeshRenderer.materials[0].EnableKeyword("_SPECULARHIGHLIGHTS_OFF");
            backgroundQuatMeshRenderer.materials[0].SetFloat("_SpecularHighlights", 0f);

        }
    }
    bool ToogleCreateOrSnapshot = true;
    int ScreenRepeatIndex = 0;
    // Update is called once per frame
    void Update()
    {
        if (ToogleCreateOrSnapshot)
        {
            LoadImageWithIndex(BgIndex);
            GenerateCar(prefabIndex);
            ToogleCreateOrSnapshot = false;
        }
        else
        {
            PlaceAndRotateCar(newObj);
            if (HasDistractorObjects)
            {
                PlaceDistractorObjectsNearVehicle();
            }
            if (HasDistractorVehicles)
            {
                PlaceDistractorVehiclesNearVehicle();
            }
            TakeSnapshot();
            countImages++;

            ScreenRepeatIndex++;
            if (ScreenRepeatIndex >= 3)
            {
                DeleteCar();
                prefabIndex++;
                ToogleCreateOrSnapshot = true;
                ScreenRepeatIndex =0;
            }
        }
        if (prefabIndex >= prefabs.Length)
        {
            prefabIndex = 0;
            BgIndex++;
        }
    }
    void DeleteCar()
    {
        System.GC.Collect();
        Resources.UnloadUnusedAssets();
        Destroy(newObj);
        System.GC.Collect();
    }
    void TakeSnapshot()
    {
        RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
        camerax.targetTexture = rt;
        (float minx, float maxx, float miny, float maxy) = GetBBoxMinMax2D(newObj);

        Rect rectBbox = Rect.MinMaxRect(minx, miny, maxx, maxy);
        resWidthBbox = Mathf.RoundToInt((maxx - minx));
        resHeightBbox = Mathf.RoundToInt((maxy - miny));
        Texture2D screenShotCropped = new Texture2D(resWidthBbox, resHeightBbox, TextureFormat.RGB24, false);
        camerax.Render();
        RenderTexture.active = rt;
        screenShotCropped.ReadPixels(rectBbox, 0, 0);
        camerax.targetTexture = null;
        RenderTexture.active = null;
        var mainCamera = GetComponent<Camera>();
        var depth = 24;
        var format = RenderTextureFormat.Default;
        var readWrite = RenderTextureReadWrite.Default;
        var renderRT = new RenderTexture(resWidth, resHeight, depth, format, readWrite);
        renderRT.Create();
        RenderTexture.active = renderRT;
        RenderTexture.active = null;
        Destroy(rt);
        Destroy(renderRT);
        byte[] bytesCropped = screenShotCropped.EncodeToJPG();
        Destroy(screenShotCropped);
        (string filename_save, string filename_save_cropped) = ScreenShotNames(countImages);
        System.IO.File.WriteAllBytes(filename_save_cropped, bytesCropped);
        Debug.Log(string.Format("Took screenshot to: {0}", filename_save));

    }
    private (string, string) ScreenShotNames(int countImages)
    {
        string filename = $"image_{countImages.ToString().PadLeft(7, '0')}_bg_{BgIndex.ToString().PadLeft(2, '0')}_carid_{prefabIndex.ToString().PadLeft(4, '0')}_alpha_{alpha.ToString()}_beta_{beta.ToString()}";
        var filenameExtension = System.IO.Path.GetExtension(filename);
        if (filenameExtension == "")
            filenameExtension = ".jpg";
        var filename_save = Path.Combine("captures/img", filename) + filenameExtension;
        var filename_save_cropped = Path.Combine("captures/img_cropped", filename) + filenameExtension;
        return (filename_save, filename_save_cropped);
    }
    private void PlaceAndRotateCar(GameObject go)
    {
        float newX, newY, newZ;
        newZ = Random.Range(0f, 22f); // bei 20 ist y 16 und x 28 // -4 ist y 12 und x 20
        newX = Random.Range(-(newZ / 3f + 21.33f), newZ / 3f + 21.33f);
        newY = Random.Range(-(newZ / 4f + 12.67f), newZ / 4f + 12.67f);
        newPos = new Vector3(newX, newY, newZ);
        go.transform.position = newPos;
        go.transform.eulerAngles = new Vector3(0, 0, 0);
        camerax.transform.position = new Vector3(newPos.x, newPos.y, -10);
        alpha = Random.Range(0, 360);
        beta1 = Random.Range(0, 90);
        beta2 = Random.Range(0, 90);
        beta = Mathf.Min(beta1, beta2);
        go.transform.RotateAround(newPos, new Vector3(0, 1, 0), alpha);
        go.transform.RotateAround(newPos, new Vector3(-1, 0, 0), beta);
    }
    private void PlaceDistractorVehiclesNearVehicle()
    {
        Destroy(DistractorVehicles[0]);
        Destroy(DistractorVehicles[1]);
        Destroy(DistractorVehicles[2]);

        // if (Random.Range(0f, 1.0f) < 0.5)
        if (true)
        {
            indexDistractorVehicle = Random.Range(0, prefabs.Length);
            GameObject prefabDistractorVehicle = prefabs[indexDistractor];
            if (Random.Range(0f, 1.0f) < 0.5) // mach den distractor entweder links oder rechts
            {
                newPosDistractorVehicle = newObj.transform.position + new Vector3(Random.Range(-5f, -3f), 0, Random.Range(0.5f, 5f));
                // newPosDistractorVehicle = newObj.transform.position + new Vector3(0, 0, 0);//Random.Range(-4f, -4f));
            }
            else
            {
                newPosDistractorVehicle = newObj.transform.position + new Vector3(Random.Range(3f, 5f), 0, Random.Range(0.5f, 5f));
                // newPosDistractorVehicle = newObj.transform.position + new Vector3(0, 0, 0);// Random.Range(-4f, -4f));
            }
            newRotDistractorVehicle.eulerAngles = new Vector3(0, 0, 0);
            prefabDistractorVehicle.GetComponent<Rigidbody>();
            DistractorVehicles[0] = Instantiate(prefabDistractorVehicle, newPosDistractorVehicle, newRotDistractorVehicle);
            // RescaleObject(DistractorVehicles[0]);
            if (DistractorVehicles[0].HasComponent<Rigidbody>())
            {
                DistractorVehicles[0].GetComponent<Rigidbody>().useGravity = false;
                Destroy(DistractorVehicles[0].GetComponent<Rigidbody>());
            }
            DistractorVehicles[0].transform.rotation = newObj.transform.rotation;
            DistractorVehicles[0].transform.RotateAround(newPosDistractorVehicle, new Vector3(0, 1, 0), Random.Range(0, 360));
            // if (newObj.transform.position.z+1 < DistractorVehicles[0].transform.position.z) // newObj is naeher an der cam als distractor
            // {

            //     RescaleObject(DistractorVehicles[0], 5);// mach distractor gross
            // }else{
            //     RescaleObject(DistractorVehicles[0], 2); // mach ditractor klein
            // }

        }

    }
    private void PlaceDistractorObjectsNearVehicle()
    {
        DestroyImmediate(Distractor);
        if (Random.Range(0f, 1.0f) < 0.5)
        { // place distractors with 50 50 chance
            indexDistractor = Random.Range(0, prefabsDistractors.Length);
            GameObject prefabDistractor = prefabsDistractors[indexDistractor];
            newPosDistractor = newObj.transform.position + new Vector3(Random.Range(-1f, 1f), Random.Range(-3f, 1f), Random.Range(-2f, -4f));
            newRotDistractor.eulerAngles = new Vector3(Random.Range(-10f, 10f), Random.Range(-10f, 10f), Random.Range(-10f, 10f));
            prefabDistractor.GetComponent<Rigidbody>();
            Distractor = Instantiate(prefabDistractor, newPosDistractor, newRotDistractor);
            if (Distractor.HasComponent<Rigidbody>())
            {
                Distractor.GetComponent<Rigidbody>().useGravity = false;
                DestroyImmediate(Distractor.GetComponent<Rigidbody>());
            }
            alphaDistractor = Random.Range(0, 360);
            Distractor.transform.RotateAround(newPosDistractor, new Vector3(0, 1, 0), alphaDistractor);
        }


    }
    Vector3 newRotEuler;
    void GenerateCar(int prefabIndex)
    {

        DeleteCar();
        GameObject prefab = prefabs[prefabIndex];
        newPos = new Vector3(0, 0, 0);
        newRot.eulerAngles = new Vector3(0, 90, 0);
        prefab.GetComponent<Rigidbody>();
        newObj = Instantiate(prefab, newPos, newRot);
        RescaleObject(newObj);
        if (newObj.HasComponent<Rigidbody>())
        {
            newObj.GetComponent<Rigidbody>().useGravity = false;
            Destroy(newObj.GetComponent<Rigidbody>());
        }
    }

    private void RescaleObject(GameObject go, int size = 5)
    {
        float max = -Mathf.Infinity;
        Bounds b = new Bounds(Vector3.zero, Vector3.zero);
        MeshFilter[] meshFilter = go.GetComponentsInChildren<MeshFilter>();
        MeshFilter[] mf = new MeshFilter[meshFilter.Length];
        for (int j = 0; j < meshFilter.Length; j++)
        {
            mf[j] = meshFilter[j];
            b.Encapsulate(mf[j].gameObject.GetComponent<Renderer>().bounds);
        }
        max = Mathf.Max(Mathf.Max(b.size.x, b.size.y), b.size.z);
        scaleFactor = size / max;
        go.transform.localScale = go.transform.localScale * scaleFactor;

    }

    private (float, float, float, float) GetBBoxMinMax2D(GameObject go)
    {
        Vector3[] vertices;
        Vector3[] vertices_trans;
        Vector3[] screenPos;
        float minx = 2000;
        float maxx = 0;
        float miny = 2000;
        float maxy = 0;
        // foreach (Transform child in newObj.transform){
        MeshFilter[] meshFilter = go.GetComponentsInChildren<MeshFilter>();

        MeshFilter[] mf = new MeshFilter[meshFilter.Length];
        // foreach (MeshFilter mf in meshFilter) {
        for (int j = 0; j < meshFilter.Length; j++)
        {
            mf[j] = meshFilter[j];
            vertices = mf[j].sharedMesh.vertices;
            // vertices = mf[j].mesh.vertices;
            screenPos = new Vector3[vertices.Length];
            vertices_trans = new Vector3[vertices.Length];
            for (int i = 0; i < vertices.Length; i++)
            {
                vertices_trans[i] = mf[j].gameObject.transform.TransformPoint(vertices[i]);
                screenPos[i] = camerax.WorldToScreenPoint(vertices_trans[i]);
                // screenPos[i].y = 1080 - screenPos[i].y;

                minx = (screenPos[i].x < minx) ? screenPos[i].x : minx;
                maxx = (screenPos[i].x > maxx) ? screenPos[i].x : maxx;
                miny = (screenPos[i].y < miny) ? screenPos[i].y : miny;
                maxy = (screenPos[i].y > maxy) ? screenPos[i].y : maxy;
            }
        }
        float margin = Random.Range(15, 35) * scaleFactor;
        minx -= Random.Range(5, 30) * scaleFactor; ;
        maxx += Random.Range(5, 30) * scaleFactor; ;
        miny -= Random.Range(5, 30) * scaleFactor; ;
        maxy += Random.Range(5, 30) * scaleFactor; ;
        minx = Mathf.RoundToInt(minx);//*1920);
        maxx = Mathf.RoundToInt(maxx);//*1920);
        miny = Mathf.RoundToInt(miny);//*1080);
        maxy = Mathf.RoundToInt(maxy);//*1080);
        return (minx, maxx, miny, maxy);

    }

    private Camera CreateHiddenCamera(string name)
    {
        var go = new GameObject(name, typeof(Camera));
        go.hideFlags = HideFlags.HideAndDontSave;
        go.transform.parent = transform;

        var newCamera = go.GetComponent<Camera>();
        return newCamera;
    }
}
