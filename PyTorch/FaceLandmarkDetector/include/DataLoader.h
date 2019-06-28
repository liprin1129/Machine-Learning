/*class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
*/

#ifndef __IMAGE_DATA_LOADER_H__
#define __IMAGE_DATA_LOADER_H__

#include <torch/torch.h>

#include <iostream>
#include <fstream>
#include <string>
#include <experimental/filesystem>

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // resize

namespace filesystem = std::experimental::filesystem;

class CustomDataset: public torch::data::Dataset<CustomDataset> {
    private:
        //torch::Tensor _states, _labels; // Return Tensors
        std::vector<std::tuple<std::string, std::vector<int>>> _dataset;     // Return dataset vector (image, label) string
        std::string _loc;

    public:
        explicit CustomDataset(const std::string& loc_states);// { readCSV(loc_states); };

        torch::data::Example<> get(size_t index) override;
        torch::Tensor read_data(const std::string &loc);

        void readCSV(const std::string &loc);

        // Override the size method to infer the size of the data set.
        torch::optional<size_t> size() const override {
            //std::cout << _dataset.size() << std::endl;
            return _dataset.size();
        };
};

#endif // __IMAGE_DATA_LOADER_H__