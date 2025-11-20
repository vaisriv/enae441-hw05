{
    pkgs,
    perSystem,
    ...
}:
perSystem.devshell.mkShell {
    name = "enae441 hw";
    motd = ''
        {141}🚀 enae441 hw{reset} shell
        $(type -p menu &>/dev/null && menu)
    '';

    commands = [
        # python helper
        # {
        #     name = "py";
        #     category = "[python]";
        #     help = "run submission python script";
        #     command = "python ../code/submission.py"
        # }
    ];

    packages = with pkgs; [
        # latex
        texlive.combined.scheme-full
        texlab

        # python
        (python3.withPackages (ps:
            with ps; [
                # python packages here
                matplotlib
                numpy
                scipy
                cartopy
            ]))
    ];
}
