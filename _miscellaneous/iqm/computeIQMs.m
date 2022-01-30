function scores = computeIQMs(root_dir)
    root_dir='/scratch/p288722/datasets/vision/all_I_frames';
    scores = struct('name', {}, 'brisque', {}, 'piqe', {}, 'niqe', {});
    filelist = dir(fullfile(root_dir, '**/*/*.png'));
    filelist = filelist(~[filelist.isdir]);  %remove folders from list

    %f = waitbar(0, 'Starting');
    reverseStr = '';
    n = numel(filelist);
    for k=1:n
        filepath = filelist(k);
        im = imread(append(filepath.folder, filesep, filepath.name));
        scores(k) = struct('name', filepath.name, 'brisque', brisque(im), 'piqe', piqe(im), 'niqe', niqe(im));
        %waitbar(k/n, f, sprintf('Progress: %d %%', floor(k/n*100)));
        % Display the progress
        msg = sprintf('Percent done: %3.1f', 100 * k / n);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));

        if k==5
           break
        end

    end
    save iqm_scores.mat scores
end