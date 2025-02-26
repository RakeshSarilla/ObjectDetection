package com.winit.cameraai

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.winit.cameraai.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {


    lateinit var bitmap: Bitmap
    lateinit var textureView: TextureView
    lateinit var imageView: ImageView
    lateinit var cameraManager: CameraManager
    lateinit var handler: Handler
    lateinit var cameraDevice: CameraDevice
    lateinit var model: SsdMobilenetV11Metadata1
    lateinit var imageProcessor: ImageProcessor
    var paint = Paint()
    val colors = listOf<Int>(
        Color.BLUE,
        Color.GREEN,
        Color.RED,
        Color.CYAN,
        Color.GRAY,
        Color.BLACK,
        Color.DKGRAY,
        Color.MAGENTA,
        Color.YELLOW,
        Color.LTGRAY
    )
    lateinit var lables: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        get_permission();

        lables = FileUtil.loadLabels(this, "labels.txt")
        model = SsdMobilenetV11Metadata1.newInstance(this)
        imageProcessor =
            ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()

        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        textureView = findViewById(R.id.textureView)
        imageView = findViewById(R.id.imageView)

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(p0: SurfaceTexture, p1: Int, p2: Int) {
                open_camera()
            }

            override fun onSurfaceTextureSizeChanged(p0: SurfaceTexture, p1: Int, p2: Int) {
            }

            override fun onSurfaceTextureDestroyed(p0: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(p0: SurfaceTexture) {
                bitmap = textureView.bitmap!!

                var image = TensorImage.fromBitmap(bitmap)
//                image = imageProcessor.process(image)

                val outputs = model.process(image)

                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray
                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

                var mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                var canvas = Canvas(mutableBitmap)
                val h = mutableBitmap.height
                val w = mutableBitmap.width

                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f
                var x = 0

                scores.forEachIndexed { index, fl ->
                    x = index
                    x *= 4
                    if (fl > 0.5) {
                        paint.setColor(colors.get(index))
                        paint.style = Paint.Style.STROKE
                        canvas.drawRect(
                            RectF(
                                locations.get(x + 1) * w,
                                locations.get(x) * h,
                                locations.get(x + 3) * w,
                                locations.get(x + 2) * h,
                            ), paint
                        )
                        paint.style = Paint.Style.FILL
                        canvas.drawText(
                            lables.get(
                                classes.get(index).toInt()
                            ) + " " + fl.toString(),
                            locations.get(x + 1) * w,
                            locations.get(x) * h,
                            paint
                        )


                    }
                }
                imageView.setImageBitmap(mutableBitmap)
            }
        }
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()

    }

    @SuppressLint("MissingPermission")

    fun open_camera() {
        cameraManager.openCamera(
            cameraManager.cameraIdList[0],
            object : CameraDevice.StateCallback() {
                override fun onOpened(p0: CameraDevice) {
                    cameraDevice = p0
                    var surfaceTexture = textureView.surfaceTexture
                    var surface = Surface(surfaceTexture)
                    var cameraRequest =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                    cameraRequest.addTarget(surface)

                    cameraDevice.createCaptureSession(
                        listOf(surface),
                        object : CameraCaptureSession.StateCallback() {
                            override fun onConfigured(p0: CameraCaptureSession) {
                                p0.setRepeatingRequest(cameraRequest.build(), null, null)
                            }

                            override fun onConfigureFailed(p0: CameraCaptureSession) {
                            }

                        },
                        handler
                    )

                }

                override fun onDisconnected(p0: CameraDevice) {
                }

                override fun onError(p0: CameraDevice, p1: Int) {
                }

            }, handler

        )
    }

    fun get_permission() {
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)

        }

    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            get_permission()

        }
    }

    private fun configureTransform(viewWidth: Int, viewHeight: Int, previewSize: Size) {

        val matrix = Matrix()
        val viewRect = android.graphics.RectF(0f, 0f, viewWidth.toFloat(), viewHeight.toFloat())
        val bufferRect = android.graphics.RectF(0f, 0f, previewSize.height.toFloat(), previewSize.width.toFloat())
        val centerX = viewRect.centerX()
        val centerY = viewRect.centerY()

        bufferRect.offset(centerX - bufferRect.centerX(), centerY - bufferRect.centerY())

        matrix.setRectToRect(viewRect, bufferRect, Matrix.ScaleToFit.FILL)
        textureView.setTransform(matrix)
    }



//    private fun open_camera() {
//        cameraManager.openCamera(
//            cameraManager.cameraIdList[0],
//            object : CameraDevice.StateCallback() {
//                override fun onOpened(p0: CameraDevice) {
//                    cameraDevice = p0
//                    val characteristics = cameraManager.getCameraCharacteristics(cameraDevice.id)
//                    val map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
//                    val previewSize = map?.getOutputSizes(SurfaceTexture::class.java)?.get(0) ?: Size(1080, 1920)
//
//                    val surfaceTexture = textureView.surfaceTexture
//                    surfaceTexture?.setDefaultBufferSize(previewSize.width, previewSize.height)
//
//                    configureTransform(textureView.width, textureView.height, previewSize)
//
//                    val surface = Surface(surfaceTexture)
//                    val cameraRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
//                    cameraRequest.addTarget(surface)
//
//                    cameraDevice.createCaptureSession(
//                        listOf(surface),
//                        object : CameraCaptureSession.StateCallback() {
//                            override fun onConfigured(p0: CameraCaptureSession) {
//                                p0.setRepeatingRequest(cameraRequest.build(), null, null)
//                            }
//
//                            override fun onConfigureFailed(p0: CameraCaptureSession) {}
//                        },
//                        handler
//                    )
//                }
//
//                override fun onDisconnected(p0: CameraDevice) {}
//                override fun onError(p0: CameraDevice, p1: Int) {}
//            },
//            handler
//        )
//    }
}
