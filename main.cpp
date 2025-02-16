#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <vector>
#include <cmath>


using namespace cv;
using namespace std;

//Distanza euclidea √((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)
inline float euclideanDistance(const Vec3f& a, const Vec3f& b) {
    return sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                (a[1] - b[1]) * (a[1] - b[1]) +
                (a[2] - b[2]) * (a[2] - b[2]));
}


void meanShift(const Mat& input, Mat& output, float radius, int max_iter = 15, float epsilon = 1e-3) {
    //Conversione in floating point perchè sennò con gli interi impazzisco
    Mat data;
    input.convertTo(data, CV_32FC3);

    // Converto immagine in vettore dove ogni elemento contiene i colori RGB
    vector<Vec3f> points;
    for (int y = 0; y < data.rows; ++y) {
        for (int x = 0; x < data.cols; ++x) {
            points.push_back(data.at<Vec3f>(y, x));
        }
    }

    // Creo vettore con i punti spostati
    vector<Vec3f> shiftedPoints(points.size());

    // Cicla su ogni pixel e lo prendo come punto di partenza
    for (size_t i = 0; i < points.size(); ++i) {
        Vec3f currentPoint = points[i];
        // Ciclo sul punto finchè non raggiunge le max iterazioni
        for (int iter = 0; iter < max_iter; ++iter) {
            Vec3f newPoint = Vec3f(0, 0, 0);
            float totalWeight = 0;
            //cicla sugli altri punti dell'immagine
            for (const auto& neighbor : points) {
                // calcola la distanza euclidea
                float distance = euclideanDistance(currentPoint, neighbor);
                //aggiorno i pesi se minore del mio raggio
                if (distance < radius) {
                    float weight = (distance < radius) ? 1.0 : 0.0;
                    newPoint += weight * neighbor;
                    totalWeight += weight;
                }
            }
            //divido per il peso totale per trovare il nuovo punto
            newPoint /= totalWeight;
            //se minore di epsilon, si è spostato poco, esco
            if (euclideanDistance(currentPoint, newPoint) < epsilon) break;
            currentPoint = newPoint;
        }
        shiftedPoints[i] = currentPoint;
    }

    // ricompongo l'immagine
    output = Mat(data.rows, data.cols, CV_32FC3);
    int index = 0;
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            output.at<Vec3f>(y, x) = shiftedPoints[index++];
        }
    }

    // Converto di nuovo in 8bit
    output.convertTo(output, CV_8UC3);
}

int main() {

    Mat input = imread("/home/edoardo/CLionProjects/MeanShiftClusterSeq/test_images/paesaggio-grande.jpg");
    Mat originalInput=imread("/home/edoardo/CLionProjects/MeanShiftClusterSeq/test_images/paesaggio-grande.jpg");
    //ridimensiono dimezzando il numero di pixel(?)

    resize(input, input, Size(), 0.5, 0.5, INTER_AREA);
    resize(originalInput, originalInput, Size(), 0.5, 0.5, INTER_AREA);

    // converto in rgb e poi in Lab
    cvtColor(input, input, COLOR_BGR2RGB);
    cvtColor(input, input, COLOR_RGB2Lab);

   
    Mat output;
    float radius = 3.0;

    auto start = std::chrono::high_resolution_clock::now();
    meanShift(input, output, radius);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    cout << "Tempo esecuzione sequenziale: " << elapsed.count() << " seconds" << endl;

    // Converto di nuovo in BGR
    cvtColor(output,output,COLOR_Lab2RGB);
    cvtColor(output, output, COLOR_RGB2BGR);


    imshow("Processata", input);
    imshow("Originale", output);
    waitKey(0);

    return 0;
}
