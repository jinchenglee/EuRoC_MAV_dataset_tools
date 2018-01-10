%==========================================================================
% Actually reads YAML file and transforms it using several mechanisms:
%
%   - Transforms mappings and lists into Matlab structs and cell arrays,
%     for timestamps uses DateTime class, performs all imports (when it
%     finds a struct field named 'import' it opens file(s) named in the
%     field content and substitutes the filename by their content.
%   - Deflates outer imports into inner imports - see deflateimports(...)
%     for details.
%   - Merges imported structures with the structure from where the import
%     was performed. This is actually the same process as inheritance with
%     the difference that parent is located in a different file.
%   - Does inheritance - see doinheritance(...) for details.
%   - Makes matrices from cell vectors - see makematrices(...) for details.
%
% Parameters:
%   filename         ... name of an input yaml file
%   nosuchfileaction ... Determines what to do if a file to read is missing
%                        0 or not present - missing file will only throw a
%                                           warning
%                        1                - missing file throws an
%                                           exception and halts the process
%   makeords         ... Determines whether to convert cell array to
%                        ordinary matrix whenever possible (1).
%   dictionary       ... Dictionary of of labels that will be replaced,
%                        struct is expected
function result = ReadYaml(filename, nosuchfileaction, makeords, treatasdata, dictionary)
    if ~exist('nosuchfileaction','var')
        nosuchfileaction = 0;
    end;
    if ~ismember(nosuchfileaction,[0,1])
        error('nosuchfileexception parameter must be 0,1 or missing.');
    end;
    if ~exist('makeords','var')
        makeords = 0;
    end;
    if ~ismember(makeords,[0,1])
        error('makeords parameter must be 0,1 or missing.');
    end;    
    if(~exist('treatasdata','var'))
        treatasdata = 0;
    end;
    if ~ismember(treatasdata,[0,1])
        error('treatasdata parameter must be 0,1 or missing.');
    end; 

    
    ry = ReadYamlRaw(filename, 1, nosuchfileaction, treatasdata);
    %<<JC>> ry = deflateimports(ry);
    %<<JC>> if iscell(ry) && ...
    %<<JC>>     length(ry) == 1 && ...
    %<<JC>>     isstruct(ry{1}) && ...
    %<<JC>>     length(fields(ry{1})) == 1 && ...
    %<<JC>>     isfield(ry{1},'import')        
    %<<JC>>     ry = ry{1};
    %<<JC>> end;
    %<<JC>> ry = mergeimports(ry);    
    %<<JC>> ry = doinheritance(ry);
    %<<JC>> ry = makematrices(ry, makeords);    
    %<<JC>> if exist('dictionary','var')
    %<<JC>>     ry = dosubstitution(ry, dictionary);
    %<<JC>> end;
    
    result = ry;
    clear global nsfe;
end
