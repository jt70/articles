package security_article

import ai.djl.Application
import ai.djl.engine.Engine
import ai.djl.modality.cv.Image
import ai.djl.modality.cv.ImageFactory
import ai.djl.modality.cv.output.DetectedObjects
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject
import ai.djl.modality.cv.output.Rectangle
import ai.djl.repository.zoo.Criteria
import ai.djl.repository.zoo.ModelZoo
import ai.djl.training.util.ProgressBar
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.selects.select
import org.bytedeco.javacv.*
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Graphics2D
import java.awt.image.BufferedImage
import java.lang.System.currentTimeMillis
import kotlin.system.exitProcess

fun annotateImage(javaImage: BufferedImage, frameRate: Int, objects: DetectedObjects?) {
    val g2d: Graphics2D = javaImage.createGraphics()
    val font = g2d.font
    g2d.font = font.deriveFont(javaImage.height * 0.03f)
    g2d.drawString(String.format("FPS: %d", frameRate), 5, (javaImage.height * 0.03f).toInt())
    if (objects != null && objects.numberOfObjects > 0) {
        val width = javaImage.width
        val height = javaImage.height

        for (objectIndex in 0..<objects.numberOfObjects) {
            val detectedObject = objects.item<DetectedObject>(objectIndex)
            val box: Rectangle = detectedObject.boundingBox.bounds

            val objectInfo = String.format("%s %.1f%%", detectedObject.className, detectedObject.probability * 100)
            g2d.color = Color.WHITE
            g2d.drawString(objectInfo, (box.x * width).toInt(), (box.y * height).toInt() - 5)

            g2d.color = Color.RED
            g2d.stroke = BasicStroke(2f)
            g2d.drawRect(
                (box.x * width).toInt(),
                (box.y * height).toInt(),
                (box.width * width).toInt(),
                (box.height * height).toInt()
            )
            g2d.color = Color(0, 0, 255, 30)
            g2d.fillRect(
                (box.x * width).toInt(),
                (box.y * height).toInt(),
                (box.width * width).toInt(),
                (box.height * height).toInt()
            )
        }
    }
}

suspend fun startFrameGrabber(
    displayChannel: Channel<BufferedImage>, objectDetectionChannel: Channel<BufferedImage>
) {
    //val grabber = FFmpegFrameGrabber("rtsp://jason:my_password@192.168.1.212:554/stream0");
    val grabber = FrameGrabber.createDefault(0)
    grabber.start()
    val javaFrameConverter = Java2DFrameConverter()
    while (true) {
        val cameraFrame: Frame = grabber.grab()
        val bufferedImage = javaFrameConverter.getBufferedImage(cameraFrame)
        displayChannel.send(bufferedImage)
        objectDetectionChannel.send(bufferedImage)
    }
}

suspend fun startObjectDetector(inputChannel: Channel<BufferedImage>, outputChannel: Channel<DetectedObjects>) {
    ModelZoo.listModels().forEach { println(it) }
    val criteria = Criteria.builder().optApplication(Application.CV.OBJECT_DETECTION)
        .setTypes(Image::class.java, DetectedObjects::class.java).optArtifactId("yolov5s")
        .optEngine(Engine.getDefaultEngineName()).optProgress(ProgressBar()).build()
    val predictor = criteria.loadModel().newPredictor()
    while (true) {
        val javaImage = inputChannel.receive()
        val djlImage = ImageFactory.getInstance().fromImage(javaImage)
        val objects: DetectedObjects = predictor.predict(djlImage)
        outputChannel.send(objects)
    }
}

suspend fun startDisplay(
    images: Channel<BufferedImage>, objects: Channel<DetectedObjects>, recorderChannel: Channel<BufferedImage>
) {
    val frame = CanvasFrame("Kotlin object detection")
    var lastTime: Long = currentTimeMillis()
    var detectedObjects: DetectedObjects? = null
    while (frame.isVisible) {
        select {
            images.onReceive { value ->
                val time = currentTimeMillis()
                val frameRate = 1000 / (time - lastTime)
                lastTime = time
                annotateImage(value, frameRate.toInt(), detectedObjects)
                frame.showImage(value)
                recorderChannel.send(value)
            }
            objects.onReceive { value ->
                detectedObjects = value
                Unit
            }
        }
    }
    exitProcess(0)
}

suspend fun startRecorder(secondsPerFile: Int, recorderChannel: Channel<BufferedImage>) {
    val converter = Java2DFrameConverter()
    var recorder: FrameRecorder? = null
    var startTime = 0L
    while (true) {
        val frame = recorderChannel.receive()
        val currentTime = currentTimeMillis()
        if (currentTime - startTime > secondsPerFile * 1000) {
            recorder?.stop()
            val filename = "output-${startTime / 1000}.mp4"
            recorder = FrameRecorder.createDefault(filename, frame.width, frame.height)
            recorder.start()
            startTime = currentTimeMillis()
        }
        recorder?.record(converter.convert(frame))
    }
}
fun main() = runBlocking {
    val displayChannel = Channel<BufferedImage>(Channel.CONFLATED)
    val objectDetectionChannel = Channel<BufferedImage>(Channel.CONFLATED)
    val detectedObjectsChannel = Channel<DetectedObjects>(Channel.CONFLATED)
    val recorderChannel = Channel<BufferedImage>(Channel.CONFLATED)

    launch(Dispatchers.Default) {
        startFrameGrabber(displayChannel, objectDetectionChannel)
    }
    launch(Dispatchers.Default) {
        startObjectDetector(objectDetectionChannel, detectedObjectsChannel)
    }
    launch(Dispatchers.Default) {
        startDisplay(displayChannel, detectedObjectsChannel, recorderChannel)
    }
    launch(Dispatchers.Default) {
        startRecorder(30, recorderChannel)
    }
    println("System started.")
}
