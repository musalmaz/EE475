import numpy as np
import cv2


class SeamCarver:
    def __init__(self, filename, output_width, output_height, SAVE_STEPS, object_mask=''):
        # initialize parameter
        self.filename = filename
        self.output_height = output_height
        self.output_width = output_width
        self.SAVE_STEPS = SAVE_STEPS
        # read in image and store as np.float64 format
        self.input_image = cv2.imread(filename).astype(np.float64)
        self.input_height, self.input_width = self.input_image.shape[: 2]

        # keep tracking resulting image
        self.output_image = np.copy(self.input_image)

        # object removal --> self.object = True
        self.object = (object_mask != '')
        if self.object:
            # read in object mask image file as np.float64 format in gray scale
            self.mask = cv2.imread(object_mask, 0).astype(np.float64)

        # kernel for forward energy map calculation
        self.kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
        self.kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
        self.kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

        self.counter = 0

        # constant for covered area by protect mask or object mask
        self.constant = 1000

        # starting program
        self.start()


    def start(self):
        """
        If object mask is provided --> object removal function will be executed
        else --> seam carving function (image retargeting) will be process
        """
        if self.object:
            self.object_removal()
        else:
            self.seams_carving()
        cv2.imwrite("output.png", self.output_image)


    def seams_carving(self):
        """
        :return:

        We first process seam insertion or removal in vertical direction then followed by horizontal direction.

        If targeting height or width is greater than original ones --> seam insertion,
        else --> seam removal

        The algorithm is written for seam processing in vertical direction (column), so image is rotated 90 degree
        counter-clockwise for seam processing in horizontal direction (row)
        """

        # calculate number of rows and columns needed to be inserted or removed
        delta_row, delta_col = int(self.output_height - self.input_height), int(self.output_width - self.input_width)

        # remove column
        if delta_col < 0:
            self.seams_removal(-delta_col)
        # insert column
        elif delta_col > 0:
            self.seams_insertion(delta_col)

        # remove row
        if delta_row < 0:
            self.output_image = self.rotate_image(self.output_image, 1)
            self.seams_removal(-delta_row)
            self.output_image = self.rotate_image(self.output_image, 0)
        # insert row
        elif delta_row > 0:
            self.output_image = self.rotate_image(self.output_image, 1)
            self.seams_insertion(delta_row)
            self.output_image = self.rotate_image(self.output_image, 0)


    #Musa
    def seams_insertion(self, num_pixel):
        temp_image = np.copy(self.output_image)
        seams_record = []

        for dummy in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_backward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            seams_record.append(seam_idx)
            self.delete_seam(seam_idx)

        self.output_image = np.copy(temp_image)
        n = len(seams_record)
        for dummy in range(n):
            seam = seams_record.pop(0)
            self.add_seam(seam)
            seams_record = self.update_seams(seams_record, seam)


    def calc_energy_map(self):
        b, g, r = cv2.split(self.output_image)
        # Apply Sobel operator in both x and y directions on each color channel
        b_energy = np.absolute(cv2.Sobel(b, 6, 1, 0, ksize=3)) + np.absolute(
            cv2.Sobel(b ,6, 0, 1, ksize=3))
        g_energy = np.absolute(cv2.Sobel(g, 6, 1, 0, ksize=3)) + np.absolute(
            cv2.Sobel(g, 6, 0, 1, ksize=3))
        r_energy = np.absolute(cv2.Sobel(r, 6, 1, 0, ksize=3)) + np.absolute(
            cv2.Sobel(r, 6, 0, 1, ksize=3))

        # Combine the energies from each channel
        return b_energy + g_energy + r_energy

    #Musa
    def cumulative_map_backward(self, energy_map):
        m, n = energy_map.shape
        output = np.copy(energy_map)

        for row in range(1, m):
            for col in range(n):
                left = max(col - 1, 0)
                right = min(col + 2, n)
                output[row, col] += np.min(output[row - 1, left:right])

        return output

    # Musa
    # def cumulative_map_forward(self, energy_map):
    #     matrix_x = self.calc_neighbor_matrix(self.kernel_x)
    #     matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
    #     matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)
    #
    #     m, n = energy_map.shape
    #     output = np.copy(energy_map)
    #     for row in range(1, m):
    #         for col in range(n):
    #             e_up = output[row - 1, col] + matrix_x[row - 1, col]
    #             if col == 0:
    #                 e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
    #                 output[row, col] = energy_map[row, col] + min(e_right, e_up)
    #             elif col == n - 1:
    #                 e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
    #                 output[row, col] = energy_map[row, col] + min(e_left, e_up)
    #             else:
    #                 e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
    #                 e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
    #                 output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
    #     return output
    def cumulative_map_forward(self, energy_map):
        matrix_x = self.calc_neighbor_matrix(self.kernel_x)
        matrix_y_left = self.calc_neighbor_matrix(self.kernel_y_left)
        matrix_y_right = self.calc_neighbor_matrix(self.kernel_y_right)

        m, n = energy_map.shape
        output = np.copy(energy_map)

        for row in range(1, m):
            for col in range(n):
                e_up = output[row - 1, col] + matrix_x[row - 1, col]
                e_left = output[row - 1, max(col - 1, 0)] + matrix_x[row - 1, max(col - 1, 0)] + matrix_y_left[
                    row - 1, max(col - 1, 0)]
                e_right = output[row - 1, min(col + 1, n - 1)] + matrix_x[row - 1, min(col + 1, n - 1)] + \
                          matrix_y_right[row - 1, min(col + 1, n - 1)]

                output[row, col] = energy_map[row, col] + min(e_up, e_left, e_right)

        return output

    # Musa
    def calc_neighbor_matrix(self, kernel):
        # Splitting the image into its color channels
        channels = cv2.split(self.output_image)

        # Applying the filter to each channel and summing the absolute values
        output = sum(np.absolute(cv2.filter2D(channel, -1, kernel)) for channel in channels)

        return output

    def seams_removal(self, num_pixel):
        for _ in range(num_pixel):
            energy_map = self.calc_energy_map()
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            self.delete_seam(seam_idx)

    # def find_seam(self, cumulative_map):
    #     m, n = cumulative_map.shape
    #     output = np.zeros((m,), dtype=np.uint32)
    #     output[-1] = np.argmin(cumulative_map[-1])
    #     for row in range(m - 2, -1, -1):
    #         prv_x = output[row + 1]
    #         if prv_x == 0:
    #             output[row] = np.argmin(cumulative_map[row, : 2])
    #         else:
    #             output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
    #     return output

    def find_seam(self, cumulative_map):
        m, n = cumulative_map.shape
        output = np.zeros(m, dtype=np.uint32)

        # Start from the bottom row
        output[-1] = np.argmin(cumulative_map[-1])

        # Move upwards to find the minimum energy seam
        for row in range(m - 2, -1, -1):
            prv_x = output[row + 1]
            col_range = slice(max(prv_x - 1, 0), min(prv_x + 2, n))
            output[row] = np.argmin(cumulative_map[row, col_range]) + col_range.start

        return output

    def delete_seam(self, seam_idx):
        m, n = self.output_image.shape[: 2]
        output = np.zeros((m, n - 1, 3))
        trace = np.copy(self.output_image)
        if self.SAVE_STEPS:
            for r in range(m):
                c = seam_idx[r]
                # Set the pixel to red in the output image
                trace[r, c] = [0, 0, 255]  # RGB value for red
            filename = f"images_tracked/{self.counter}.png"  # Format the filename string
            cv2.imwrite(filename, trace)  # Save the image
            # Increment the counter if you're doing this in a loop
            self.counter += 1
        for row in range(m):
            col = seam_idx[row]
            output[row, :, 0] = np.delete(self.output_image[row, :, 0], [col])
            output[row, :, 1] = np.delete(self.output_image[row, :, 1], [col])
            output[row, :, 2] = np.delete(self.output_image[row, :, 2], [col])
        self.output_image = np.copy(output)


    def rotate_image(self, image,clockwise):
        m, n, ch = image.shape
        output = np.zeros((n, m, ch))
        if not clockwise:
            image_flip = np.fliplr(image)
            for c in range(ch):
                for row in range(m):
                    output[:, row, c] = image_flip[row, :, c]
        else:
            for c in range(ch):
                for row in range(m):
                    output[:, m - 1 - row, c] = image[row, :, c]
        return  output

    # Musa
    def rotate_mask(self, mask, clockwise):
        m, n = mask.shape
        output = np.zeros((n, m))
        if clockwise:
            image_flip = np.fliplr(mask)
            for row in range(m):
                output[:, row] = image_flip[row, : ]
        else:
            for row in range(m):
                output[:, m - 1 - row] = mask[row, : ]
        return output

    # İsmail
    def object_removal(self):
        """
        :return:

        Object covered by mask will be removed first and seam will be inserted to return to original image dimension
        """
        rotate = False
        object_height, object_width = self.get_object_dimension()
        if object_height < object_width:
            self.output_image = self.rotate_image(self.output_image, 1)
            self.mask = self.rotate_mask(self.mask, 1)
            rotate = True

        while len(np.where(self.mask[:, :] > 0)[0]) > 0:
            energy_map = self.calc_energy_map()
            energy_map[np.where(self.mask[:, :] > 0)] *= -self.constant
            cumulative_map = self.cumulative_map_forward(energy_map)
            seam_idx = self.find_seam(cumulative_map)
            self.delete_seam(seam_idx)
            self.delete_seam_on_mask(seam_idx)

        if not rotate:
            num_pixels = self.input_width - self.output_image.shape[1]
        else:
            num_pixels = self.input_height - self.output_image.shape[1]

        self.seams_insertion(num_pixels)
        if rotate:
            self.output_image = self.rotate_image(self.output_image, 0)



    # İsmail
    def add_seam(self, seam_idx):
        m, n = self.output_image.shape[: 2]
        output = np.zeros((m, n + 1, 3))
        for row in range(m):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(self.output_image[row, col: col + 2, ch])
                    output[row, col, ch] = self.output_image[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = self.output_image[row, col:, ch]
                else:
                    p = np.average(self.output_image[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = self.output_image[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = self.output_image[row, col:, ch]
        self.output_image = np.copy(output)

    # İsmail
    def update_seams(self, remaining_seams, current_seam):
        output = []
        for seam in remaining_seams:
            seam[np.where(seam >= current_seam)] += 2
            output.append(seam)
        return output

    # İsmail
    def delete_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n - 1))
        for row in range(m):
            col = seam_idx[row]
            output[row, : ] = np.delete(self.mask[row, : ], [col])
        self.mask = np.copy(output)

    # İsmail
    def add_seam_on_mask(self, seam_idx):
        m, n = self.mask.shape
        output = np.zeros((m, n + 1))
        for row in range(m):
            col = seam_idx[row]
            if col == 0:
                p = np.average(self.mask[row, col: col + 2])
                output[row, col] = self.mask[row, col]
                output[row, col + 1] = p
                output[row, col + 1: ] = self.mask[row, col: ]
            else:
                p = np.average(self.mask[row, col - 1: col + 1])
                output[row, : col] = self.mask[row, : col]
                output[row, col] = p
                output[row, col + 1: ] = self.mask[row, col: ]
        self.mask = np.copy(output)

    # Musa
    def get_object_dimension(self):
        rows, cols = np.where(self.mask > 0)
        height = np.amax(rows) - np.amin(rows) + 1
        width = np.amax(cols) - np.amin(cols) + 1
        return height, width


    def save_result(self, filename):
        cv2.imwrite(filename, self.output_image.astype(np.uint8))


seam_carving = SeamCarver("test.jpg", 340, 480, 1)
