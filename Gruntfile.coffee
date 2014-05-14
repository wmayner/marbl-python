module.exports = (grunt) ->
  grunt.initConfig
    cfg:
      docDir: "docs"
      src: "marbl.py"
      testDir: "test"

    shell:
      test:
        command: "coverage run --source <%= cfg.src %> -m py.test"
        options:
          stdout: true
      coverageHTML:
        command: "coverage html"
        options:
          stdout: true
      buildDocs:
        command: [
          "cd docs"
          "make html"
          "cp _static/* _build/html/_static"
        ].join "&&"
        options:
          stdout: true
          failOnError: true
      uploadDocs:
        command: "python setup.py upload_docs --upload_dir docs/_build/html"
        options:
          stdout: true
          failOnError: true
      openDocs:
        command: "open docs/_build/html/index.html"
      openCoverage:
        command: "open htmlcov/index.html"

    watch:
      docs:
        files: [
          "Gruntfile.coffee"
          "**/*.rst"
          "<%= cfg.src %>"
          "<%= cfg.docDir %>/**/*"
          "!<%= cfg.docDir %>/_build/**/*"
        ]
        tasks: ["shell:buildDocs"]
      test:
        files: [
          "conftest.py"
          ".coveragerc"
          "pytest.ini"
          ".pythonrc.py"
          "<%= cfg.src %>"
          "<%= cfg.testDir %>/**/*"
          "Gruntfile.coffee"
        ]
        tasks: ["shell:test", "shell:coverageHTML"]

  # Load NPM Tasks
  grunt.loadNpmTasks "grunt-contrib-watch"
  grunt.loadNpmTasks "grunt-shell"

  # Custom Tasks
  grunt.registerTask "default", [
    "watch:test"
  ]
  grunt.registerTask "docs", [
    "shell:openDocs"
    "watch:docs"
  ]
  grunt.registerTask "uploaddocs", [
    "shell:buildDocs"
    "shell:uploadDocs"
  ]
  grunt.registerTask "test", [
    "shell:test"
    "shell:coverageHTML"
    "shell:openCoverage"
    "watch:test"
  ]
