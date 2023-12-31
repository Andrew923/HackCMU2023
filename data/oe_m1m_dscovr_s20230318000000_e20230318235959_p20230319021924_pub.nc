CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230318000000_e20230318235959_p20230319021924_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-19T02:19:24.486Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-18T00:00:00.000Z   time_coverage_end         2023-03-18T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxo A�  T          @�
=�q�@���?��A'�CL��q�@<��@FffB�C�                                    Bxo P&  �          @�33�|(�@xQ�?�@�\)Ch��|(�@?\)@!�A؏\C                                    Bxo ^�  
(          @�\)�_\)@��\?��RA�z�C���_\)@Q�@w�B)�RC�R                                    Bxo mr  
�          @��R�j=q@w
=@�A��\Cp��j=q@	��@tz�B(
=C��                                    Bxo |  
�          @�\)�^�R@vff@��A�ffC��^�R?�(�@��HB6�C�                                     Bxo ��  T          @�{�_\)@a�@0  A�(�C���_\)?�G�@��RB?��C�f                                    Bxo �d  T          @��H�P  @�
=?�A��B�#��P  @+�@j�HB$\)C��                                    Bxo �
  T          @����8Q�@�p�?���A���B�8R�8Q�@8��@l(�B(�\C�f                                    Bxo ��  T          @�Q��:=q@��R?�  AS33B�#��:=q@Fff@\(�B��C�                                    Bxo �V  
Z          @����X��@��H?�
=Apz�C�f�X��@,��@X��BQ�C�                                    Bxo ��  �          @����c�
@w�?У�A�  C�H�c�
@��@\��B{C�                                    Bxo �  T          @�
=��=q@^�R?h��AC}q��=q@p�@(Q�A�33C�f                                    Bxo �H  
�          @��R��ff@Tz�?E�A�C����ff@��@�A�\)CL�                                    Bxo ��  T          @�  �G�@�Q�>u@�RB�L��G�@l(�@'
=A�
=C5�                                    Bxo!�  "          @���]p�@��R>�=q@3�
Ck��]p�@Z�H@{Aי�C\)                                    Bxo!:  T          @����=q@<(�?��
A��\C��=q?�@7�A�C�)                                    Bxo!+�  T          @���n{@z=q?��\A+33C�\�n{@0��@<��B�HCaH                                    Bxo!:�  T          @���z�H@c33?���Ae��C	�)�z�H@�@B�\B(�C��                                    Bxo!I,  T          @�=q��@R�\?У�A���C�\��?�z�@HQ�B	33Cn                                    Bxo!W�  D          @��\��(�@1�@33A�{C�{��(�?�G�@Mp�B�C#�f                                    Bxo!fx  
          @�ff���@*�H@33A�\)C=q���?��@W�B��C%�                                    Bxo!u  "          @�33��@@  @
=qA�z�CaH��?���@[�Bz�C!�                                    Bxo!��  v          @�(���33@@  @=qA��C����33?�G�@h��B!=qC"�3                                    Bxo!�j  
          @��
�y��@A�@(��A�33C5��y��?�z�@vffB-�C#}q                                    Bxo!�  �          @�=q�u@<��@/\)A�CxR�u?��@x��B1G�C$�\                                    Bxo!��  �          @��\��33@1G�@#�
Aڏ\C���33?}p�@h��B#(�C&n                                    Bxo!�\  
�          @��
�}p�@p�@N{B�C�{�}p�>#�
@y��B2\)C1��                                    Bxo!�  �          @��\�z=q@1G�@3�
A��HC���z=q?Y��@vffB/z�C'�                                    Bxo!ۨ  �          @��\���H@��@?\)B ��C�q���H>�G�@s33B*�HC-��                                    Bxo!�N  �          @��
��ff@   @z�AîC����ff?c�
@R�\B
=C(�                                    Bxo!��  "          @�����R@(��@�AЏ\C�)���R?s33@]p�B
=C'B�                                    Bxo"�  T          @����{@)��@,��A�RC�3��{?O\)@l(�B#�C)                                    Bxo"@  T          @����(�@(Q�?޸RA�z�Cff��(�?��@7�A�=qC$5�                                    Bxo"$�  �          @�=q����@!�?�G�AQC\����?�p�@=qA��
C"�                                    Bxo"3�  �          @��H���
@�R?���A6{C���
?\@\)A�{C"��                                    Bxo"B2  �          @�����@p�>�ff@��
C�3���?���?�Q�A��RC��                                    Bxo"P�  	�          @��\���@333>���@VffC�����@��?�G�A�
=Cc�                                    Bxo"_~  
.          @��H����@7
=�.{���
C33����@!G�?�\)Ab=qCL�                                    Bxo"n$  �          @��\����@0  �@  ��G�C�����@/\)?Q�A
=C.                                    Bxo"|�  �          @��H���@;��u�33C�����@?\)?:�H@��
C�                                    Bxo"�p  "          @�(���(�@2�\��\)�c\)C�f��(�@E>��R@N{C=q                                    Bxo"�  �          @�{��=q@=q�����C:���=q@S33�=p���  C5�                                    Bxo"��  �          @�ff���R@$z�������C����R@\�Ϳ.{����CL�                                    Bxo"�b  T          @�����@W�����(�C�=����@o\)>�Q�@hQ�C
�
                                    Bxo"�  �          @�����@(Q��\)��{C�����@Z=q����\)C�=                                    Bxo"Ԯ  �          @�Q���p�@'��%��{CT{��p�@e�Q��33C��                                    Bxo"�T  �          @�{��@������
Cz���@E��
=q����C��                                    Bxo"��  �          @��
���@AG��Ǯ��G�C����@1G�?�G�AR�RC�R                                    Bxo# �  �          @��
��@AG��У�����C� ��@[�>L��@G�CJ=                                    Bxo#F  �          @�Q��Y��@=q�U���C�)�Y��@s33��������C�                                     Bxo#�  �          @�Q��L��@333�]p���\C
�
�L��@�ff��(��yG�B���                                    Bxo#,�  �          @���G�@5������?C �3�G�@��׿�
=��\)B�G�                                    Bxo#;8  �          @�����(�?���6ff�ffC"����(�@&ff��  ���Cٚ                                    Bxo#I�  �          @�ff��?�  ��ff�^�RC }q��@
�H�B�\��p�CE                                    Bxo#X�  �          @�p����@(������Q�C�����@W
=��
=����C)                                    Bxo#g*  �          @���e@B�\�!G�����C�q�e@z=q�����(�C�{                                    Bxo#u�  �          @��\�mp�@,���$z���=qC���mp�@h�ÿG���
C}q                                    Bxo#�v  �          @��R�z�@<���q��4�HC +��z�@����
=���RBꙚ                                    Bxo#�  �          @����\)@a��fff�(  B�Ǯ��\)@�z῜(��Qp�B���                                    Bxo#��  �          @�G��1�@QG��:=q��\CaH�1�@�녿E����B���                                    Bxo#�h  �          @�=q�J=q@+������=qC� �J=q@@��>aG�@5Cn                                    Bxo#�  �          @�=q����?�(�@=qA��C"����ü#�
@4z�A�\C4
                                    Bxo#ʹ  
Z          @��\���?�(�@Q�A�Q�C"�����<��
@333A�p�C3�\                                    Bxo#�Z  �          @�����  ?�\)@�A�\)C'c���  ��G�@z�A��C5=q                                    Bxo#�   D          @�=q��G�?�z�?�z�A�33C!���G�?��@�\A�{C-Ǯ                                    Bxo#��  
�          @�z���33?�p�?���A^{C����33?z�H@�A�ffC)�                                    Bxo$L  T          @����(�?��
?�z�Ah��C ޸��(�?G�@Q�A�z�C+O\                                    Bxo$�  �          @�G�����@   ?��
A+33CW
����?�
=?�A���C&�H                                    Bxo$%�  �          @�Q����H?���?�@�{C���H?�\)?�  A~{C$�                                    Bxo$4>  �          @������?�(�?+�@޸RC�����?�=q?˅A�=qC%ff                                    Bxo$B�  T          @�������@z�?Y��A��C�3����?�=q?�ffA�{C%=q                                    Bxo$Q�  �          @�Q����H@33>���@EC����H?���?���A^=qC"��                                    Bxo$`0  
�          @�����@   >\@xQ�C�����?�G�?�{Ac33C#�H                                    Bxo$n�  �          @��\��G�?�G�>k�@��C!����G�?��?�{A8z�C%L�                                    Bxo$}|  T          @�����=q@{��  �%C^���=q@�?p��A�C5�                                    Bxo$�"  T          @�G����@{�B�\��(�CQ����@   ?}p�A$��CxR                                    Bxo$��  �          @��H���@=q����
=C0����@33?\(�A{C=q                                    Bxo$�n  T          @�{��=q?h�ÿ��
��p�C'n��=q?ٙ�����W�
CT{                                    Bxo$�  T          @�  ��=q���g��&��C5�=��=q?�G��J�H�{C�\                                    Bxo$ƺ  T          @�\)��\)?�ff�
=����C&ٚ��\)@
=��p�����C�                                     Bxo$�`  "          @��R�a녾#�
��33�E33C6���a�?�p��fff�&�\C��                                    Bxo$�  �          @�ff�W�?��H�x���:ffC�H�W�@P  �%���Q�C�                                    Bxo$�  
�          @����g�?(���u��8�C)�)�g�@#33�<(��z�C�H                                    Bxo%R  �          @�33�n�R�h�������:�CA���n�R?���|(��3�C ��                                    Bxo%�  T          @��
�e��\(����R�C{CA� �e�?�
=�����9z�CB�                                    Bxo%�  �          @�\)�z=q?�{�X�����C���z=q@G������\Cc�                                    Bxo%-D  "          @�����(�?�z��K���C{��(�@C�
��\)��  Cu�                                    Bxo%;�  
�          @����`��?�33�s33�.33C�{�`��@e����ÅCp�                                    Bxo%J�  �          @�Q��Q�?�z���Q��K�C }q�Q�@L(��C�
�(�C��                                    Bxo%Y6  �          @�
=�Mp�@
=�n{�,�C���Mp�@{���p����\C:�                                    Bxo%g�  
�          @�
=�J=q@���p���-�
C��J=q@\)�   ��\)C ^�                                    Bxo%v�  
�          @�\)�G
=@z��u��2z�C@ �G
=@}p��ff��
=C (�                                    Bxo%�(  T          @��R��
?޸R��z��b��C\��
@o\)�8Q��  B�\)                                    Bxo%��  �          @�=q��?�����z��u��CQ���@Z�H�j=q�$33B��)                                    Bxo%�t  
�          @����*�H?xQ���p��j�
C ��*�H@N{�aG�� =qC�H                                    Bxo%�            @�G��#�
?(����\)�mC%��#�
@7
=�`���)�HC�)                                    Bxo%��  d          @���C�
@1��`���!\)C	Ǯ�C�
@�������33B�k�                                    Bxo%�f  �          @�
=�j�H@(��J=q���CW
�j�H@mp���(��|  C��                                    Bxo%�  T          @�ff�y��@n{    =#�
Cc��y��@L��?�33A��C��                                    Bxo%�  �          @�Q���  @P�׿�����(�C�{��  @g�>�\)@>�RC	ٚ                                    Bxo%�X  
�          @��׿aG�@HQ���{�W�BӀ �aG�@��R��R����B�33                                    Bxo&�  �          @��R���
@5��(��iG�B������
@����"�\�ޣ�B��                                    Bxo&�  
�          @�=q�Q�@*�H����o�B�  �Q�@���1���ffBǅ                                    Bxo&&J  �          @�\)�
=@(Q�����q��B�\)�
=@�p��0  ��(�B�k�                                    Bxo&4�  "          @�{����@7����
�ZG�B�G�����@�{��
���BО�                                    Bxo&C�  T          @���У�@J=q���\�Jp�B�3�У�@�p��Q�����Bؽq                                    Bxo&R<  �          @�33�ٙ�@7���\)�U(�B���ٙ�@�  �����B�W
                                    Bxo&`�  �          @�\)���?�(���z���Bҏ\���@��H�c�
�#Q�B�#�                                    Bxo&o�  d          @���z�@��  �`z�C8R��z�@����,�����B��                                    Bxo&~.  �          @��\�C33@Q�@5�B  C�f�C33?�33@�33BJCQ�                                    Bxo&��  "          @���-p�@`  @0��A��B�k��-p�?�\)@���BS{C
=                                    Bxo&�z  �          @�(����@O\)<�>�z�C�{���@2�\?�z�A�
=C�                                    Bxo&�   
�          @�(�����@I����G���C������@<(�?��HAQp�C��                                    Bxo&��  �          @�����\@!녿�  ���C�����\@Dz�u�"�\C��                                    Bxo&�l  
�          @������R?���%����C����R@;����
�`��C5�                                    Bxo&�  T          @�z��|(�@���7
=�\)C�f�|(�@U��{�l��CǮ                                    Bxo&�  �          @��
�p��@��G
=�33C�H�p��@\�Ϳ˅��Q�C	z�                                    Bxo&�^  
�          @����l(�@	���E��G�C�=�l(�@Y����=q��(�C	L�                                    Bxo'  �          @���.�R?�G���G��X�
CǮ�.�R@9���?\)�z�CO\                                    Bxo'�  T          @�  �.�R@녽��   C
�.�R@�\?��A���CB�                                    Bxo'P  
�          @����-p�@;�@\(�B"�C���-p�?Q�@�ffBf{C#B�                                    Bxo'-�  �          @��H�B�\@C33@I��B��C  �B�\?��@�  BS\)C ٚ                                    Bxo'<�  
�          @����Dz�@4z�@P  B�HC	n�Dz�?O\)@�\)BT\)C%(�                                    Bxo'KB  
�          @���>�R@Q�@Z�HB'\)Cff�>�R>�33@���BXffC-:�                                    Bxo'Y�  �          @����S33@w�?�z�Ax��C� �S33@(��@J=qB�CT{                                    Bxo'h�  �          @�G��L(�@�33?aG�A{B��L(�@G
=@3�
B �C�3                                    Bxo'w4  T          @��H�P��@�ff>�  @.�RB��{�P��@aG�@z�A�=qC��                                    Bxo'��  T          @��\�S33@�>\)?\C @ �S33@c33@��AîC�
                                    Bxo'��  
|          @���\(�@��H<#�
>\)C
=�\(�@b�\@33A�=qC#�                                    Bxo'�&             @�=q�^{@��׽�����C���^{@c33?��A���CaH                                    Bxo'��  v          @�G��J=q@�ff�u�(��B��H�J=q@qG�?��A��C�q                                    Bxo'�r  
8          @����Z=q@Q��  �˅C&f�Z=q@}p������dz�C��                                    Bxo'�  "          @�  �G
=@tz�ٙ����\C(��G
=@��>�
=@�=qB��\                                    Bxo'ݾ  v          @�\)�^�R@dzῴz��|��CL��^�R@s33?
=q@���C��                                    Bxo'�d  
          @�G��e@b�\?��HAX  C^��e@�R@333B  C^�                                    Bxo'�
  �          @���P��@`  @z�A���C  �P��?��@o\)B3(�Cٚ                                    Bxo(	�  T          @�=q�e�@AG�@��A�33C�)�e�?�Q�@e�B+p�C�                                    Bxo(V  T          @��\�p  @W
=?�{A�ffC
#��p  @�@C�
B��CxR                                    Bxo(&�  "          @�G���=q@K�?��\A1�C�R��=q@G�@��A�(�C޸                                    Bxo(5�  �          @�������@W
=>B�\@C:�����@5�?���A�Q�C��                                    Bxo(DH  �          @�Q��l��@k�=L��?z�C+��l��@K�?�{A���C^�                                    Bxo(R�  
�          @�  �Z=q@w
=�z�H�,��Cz��Z=q@u?��A7\)C�
                                    Bxo(a�  
�          @��\�}p�@{@(Q�A�
=C�}p�?#�
@XQ�B ��C*޸                                    Bxo(p:  �          @�(��z=q@1�@��A�C�)�z=q?�  @\(�B   C"J=                                    Bxo(~�  �          @����s33@;�@G�A���Cff�s33?Ǯ@L��Bz�C�                                    Bxo(��  
�          @�ff��Q�@-p�@�\Aʏ\C�3��Q�?��R@U�B�C"�\                                    Bxo(�,  	�          @����=q@7
=@(�A�=qC����=q?�@Tz�B�C �q                                    Bxo(��  
�          @�����H@5@33A�\)C33���H?�(�@K�B��C +�                                    Bxo(�x  "          @�����G�@W
=?�Q�AK�
CL���G�@ff@+�A�=qC�
                                    Bxo(�  "          @�p��~{@\(�?�Q�AL��C��~{@�H@.{A���C�
                                    Bxo(��  
Z          @��R��Q�@U�?8Q�@�(�C����Q�@#�
@��A�p�C�                                    Bxo(�j  
�          @������@;�?�  A+�Cff���@z�@33A�Q�Cٚ                                    Bxo(�  �          @�z���{@6ff?333@�33CW
��{@	��@   A��\C8R                                    Bxo)�  T          @�33�a�@Q��  ��z�C	��a�@}p���p�����C��                                    Bxo)\            @���s�
@`�׿�  �.�RC	aH�s�
@c33?Tz�A��C	                                    Bxo)   
�          @��H��p�@N{�u�&ffCO\��p�@6ff?�G�A�G�C�                                    Bxo).�  
�          @��H���@�Ϳ���\��C޸���@�R=u?B�\C�                                    Bxo)=N  T          @�(��p��?�(�@h��B*p�C"�p�׿0��@qG�B2z�C>p�                                    Bxo)K�  "          @������R��\)@Q�B�RC4����R��33@5B �\CIY�                                    Bxo)Z�  
�          @��_\)<�@��
BF��C3xR�_\)���H@g�B(z�CQQ�                                    Bxo)i@  "          @�ff�j�H�#�
@�  B=�
C4�H�j�H��(�@_\)B�RCP+�                                    Bxo)w�  T          @�{���    @b�\B$33C3�q��녿ٙ�@FffB�CJ                                    Bxo)��  
(          @��
�mp����\@g
=B,�CCT{�mp��(Q�@+�A�  CWY�                                    Bxo)�2  
�          @�������>8Q�@A�\)C1�����׿xQ�@��A�Q�C?xR                                    Bxo)��  
�          @�p���
=�#�
@R�\B��C4)��
=�˅@8Q�Bz�CH�f                                    Bxo)�~  
�          @�
=�~�R?���@UB�HCk��~�R�aG�@k�B+
=C733                                    Bxo)�$  
�          @������?�p�@\)A��
C!�q���=�G�@8��B�C2��                                    Bxo)��  "          @�ff�Z=q@9��@EB
ffC���Z=q?���@�33BC��C"��                                    Bxo)�p  T          @�
=��(�?��H?�Q�A��\C%Y���(�>�@�A��C2W
                                    Bxo)�  T          @�\)��(��녾����H��C:{��(���Q�
=q��p�C7�
                                    Bxo)��  "          @�Q�����?���&ff�׮C&�����?��H=u?(�C$�
                                    Bxo*
b  T          @��H����?���>�@��C'z�����?W
=?z�HA!�C+5�                                    Bxo*  �          @�{��G����R?W
=A	��C733��G��+�?
=@�G�C:��                                    Bxo*'�  
�          @�{���\��?B�\@�Q�C5Q����\���?(�@�{C8�)                                    Bxo*6T  �          @�p���zἣ�
>Ǯ@�Q�C4=q��z�W
=>�{@Z=qC6�                                    Bxo*D�  �          @����
���
�&ff�љ�C75����
<��8Q�����C3�3                                    Bxo*S�  �          @��R���R?�Q����z�C%B����R?��R>��@%C$�3                                    Bxo*bF  "          @�{��\)?������
�Q�C%#���\)?���?�R@�  C&xR                                    Bxo*p�  �          @�z����H?�G������C33C!� ���H?�Q�?�@���C"h�                                    Bxo*�  
�          @�p���=q?��#�
��\)C!0���=q?�33>�z�@>�RC \)                                    Bxo*�8  �          @��H��z�?��Ϳ����CE��z�@�H�����Q�C��                                    Bxo*��  
�          @�����
?�  �Mp���C#�����
@'��\)���
C�                                    Bxo*��  
�          @�����=q?Ǯ�����{C"���=q@=q���H�J�\Cn                                    Bxo*�*  �          @��H���=�G��C�
�\)C2�����?���)����RC!�                                     Bxo*��  T          @����p�>.{�I���  C1Ǯ��p�?У��,������C��                                    Bxo*�v  "          @��\�z�H@   �W����C��z�H@Vff����(�C}q                                    Bxo*�  T          @����dz�@��j=q�&�RC^��dz�@`������C�                                     Bxo*��  
�          @�G����\@�Q��Y���=qB�{���\@�33��G��'�
Bɮ                                    Bxo+h  T          @�  �G�@h���\(���
B��)�G�@�=q����a�B�\)                                    Bxo+  "          @�=q�P  @"�\�h���%��C�P  @|(�������Cz�                                    Bxo+ �  T          @��R�R�\?   ��G��V�HC+T{�R�\@#�
�q��(�RC�                                    Bxo+/Z  �          @�{�0��?
=q����pC(޸�0��@1G����H�933C��                                    Bxo+>   �          @���.�R>��H����q33C)�)�.�R@,(���=q�:�C^�                                    Bxo+L�  �          @�=q�U?\)��z��QffC*� �U@"�\�hQ��#ffC�=                                    Bxo+[L  �          @�녿s33�Ǯ��(�Q�CJ�s33@
=q��ffB�                                    Bxo+i�  �          @��׿�G�<��
��=qL�C2�H��G�@\)���R�m33B�Ǯ                                    Bxo+x�  T          @��þ�녿=p���ff¢p�Cp�{���?�\)��p�L�B̳3                                    Bxo+�>  �          @��ÿ�\)�+�������CN���\)?���\)z�B��                                     Bxo+��  "          @��R��p���{��G�#�C{k���p�?�(���ff��B�u�                                    Bxo+��  
(          @��R����Q���ǮC��
��?�{��Q�  B��                                     Bxo+�0  �          @�  ��  � ����=qz�C�|)��  ?\)��§8RB�\                                    Bxo+��  �          @�p��\?����\)¤� B�p��\@<(����
�_(�B³3                                    Bxo+�|  �          @�
=��=q��  ���¬#�C^&f��=q@  ���
33B��)                                    Bxo+�"  �          @�p�?
=�5���� B�C���?
=?�ff��G�B���                                    Bxo+��  T          @��>��!G�����y{C�xR>��#�
��±(�C��)                                    Bxo+�n  �          @�\)@)���<(��q��.�\C�H@)���c�
���R�m(�C���                                    Bxo,  �          @�=q?��
�B�\����H=qC�g�?��
�W
=��  p�C���                                    Bxo,�  �          @�z�?����P  ��p��?(�C�y�?����z�H��{z�C��H                                    Bxo,(`            @��
?�\�G
=���H�I�HC�˅?�\�J=q�����qC��
                                    Bxo,7  �          @��H?\��
=�H����C��?\�z���
=�n33C���                                    Bxo,E�  �          @�  ?��Ϳ\�L(��TffC�L�?���=����a��y=q@C�
                                    Bxo,TR  �          @�z�?��H@�\�����k33Bx?��H@~�R�0���p�B�B�                                    Bxo,b�  T          @�G�@녽�G����\33C��H@�@������d�HBCff                                    Bxo,q�  �          @�
=?��H��G���G��\C�\)?��H?�{������B��                                    Bxo,�D  �          @�33@P�׿@  ��Q��[�C��@P��?��R���\�P  A�Q�                                    Bxo,��  T          @�z�@p��%���W��C�8R@p��L����33�HC�o\                                    Bxo,��  �          @�  ?fff��(����#�C�&f?fff?����G���B^=q                                    Bxo,�6  �          @��\>\)������ff�C�4{>\)?Tz���{£�B��)                                    Bxo,��  �          @��=��
�������
=qC��H=��
?�{����B�W
                                    Bxo,ɂ  T          @�
=<��
�ff���\C�L�<��
?#�
��§�qB�
=                                    Bxo,�(  �          @���=q�(Q������~  C����=q<����®�3C-
=                                    Bxo,��  �          @�z�>\��\)��(��C���>\?�=q��  � B��                                     Bxo,�t  �          @��H>�녿���G�p�C��=>��?}p����RB���                                    Bxo-  �          @���Ǯ��Q�������C|��Ǯ?�
=����33B�L�                                    Bxo-�  �          @�{��=q?fff������C����=q@C33��33�D�B�                                      Bxo-!f  �          @�p���@(������S�
C	.��@w��6ff� p�B��H                                    Bxo-0  �          @��\�(��@�
��z��L{C�R�(��@l(��2�\��Q�B��                                    Bxo->�  �          @��þB�\@'
=@a�BU�B���B�\?E�@�=qB�G�B��                                    Bxo-MX  T          @��>���@1G�@�
=Ba�
B��3>���?��@�Q�B��{Bb��                                    Bxo-[�  �          @�p���
=@N{@�{B@��B��f��
=?��\@�B�aHC!H                                    Bxo-j�  T          @�p�����@]p�@��\BHp�B�\)����?�
=@��B�=qC0�                                    Bxo-yJ  �          @�����@N{@�=qBJB�=����?z�H@���B�k�C�                                    Bxo-��  �          @�
=��  @��@UBz�B�8R��  @*=q@���Bx(�B��q                                    Bxo-��  �          @�
=��@��@/\)A�G�B��=��@QG�@��BZB�
=                                    Bxo-�<  �          @�\)=#�
@��@   A���B�=#�
@xQ�@�z�B;ffB��                                    Bxo-��  T          @�ff���@fff@��A�Q�B�LͿ��@
=q@q�BSp�C.                                    Bxo-  �          @�
=��z�?�
=@Mp�B	33C)��z�=���@g�B�HC2�                                     Bxo-�.  �          @�  �a�@8Q�@^{BQ�C�a�?��@�(�BG�
C#:�                                    Bxo-��  �          @��R�{�@qG�?�z�A��
C0��{�@*=q@I��B=qC�                                    Bxo-�z  �          @����^{@�p�>�  @!G�C !H�^{@w�@
=qA�z�C��                                    Bxo-�   �          @���(�@��
��\��
=B�z��(�@���?�@�B�{                                    Bxo.�  "          @�33�z�@�{>#�
?�ffB�q�z�@���@(�A�=qB�\                                    Bxo.l  
�          @�������@^{>u@#33C�����@@��?�p�A���C�q                                    Bxo.)  
(          @�������?W
=@hQ�B%��C(8R���ÿQ�@h��B%�
C?��                                    Bxo.7�  T          @�����>��@e�B"{C0u�������@UB{CE�\                                    Bxo.F^  
�          @�G���Q�?��@S33B�HC&W
��Q�   @Z�HBQ�C:��                                    Bxo.U  T          @�\)���\?��@7�A�z�C5����\>u@S33B�C0޸                                    Bxo.c�  �          @�ff�S�
@QG�@�RA̸RC^��S�
?�p�@Z�HB&Q�C
                                    Bxo.rP  T          @�{�g�@(�@G
=B�
C
�g�?Y��@w
=B8{C&�                                    Bxo.��  T          @�  �Z�H@33@\��B�RC)�Z�H?z�@��BG�C*Q�                                    Bxo.��  �          @�  �j=q?�(�@dz�B"�C� �j=q>u@�=qB?�HC0O\                                    Bxo.�B  �          @��H�;�@ff@�\)BF33Ck��;�=�\)@�
=Bh�\C2��                                    Bxo.��  �          @��H�^{?\@�G�B;G�C\)�^{��Q�@���BK�RC9��                                    Bxo.��  �          @��
�Q�?�{@��B?(�Cu��Q녽���@���BX�
C5�{                                    Bxo.�4  �          @�33�>�R@{@��RBBG�CT{�>�R>L��@�Q�Bgz�C0#�                                    Bxo.��  
�          @��H�R�\@p�@z=qB2=qC!H�R�\>���@�\)BV{C.T{                                    Bxo.�  �          @���fff?�33@fffB%C#��fff>8Q�@��BA�C1\                                    Bxo.�&  
�          @�G���p�?޸R@5A�
=C����p�>�Q�@S�
B�HC/^�                                    Bxo/�  �          @����G�?��H@��A�ffC#���G�>���@2�\A�\)C0^�                                    Bxo/r  �          @����
=?��
@{A؏\C'�f��
=��@+�A�=qC5ff                                    Bxo/"  �          @�{���þ�\)?�G�A��C7#����ÿ�  ?�(�A�(�C?33                                    Bxo/0�  
�          @�(����?\(�?���A=��C*�\���>��R?�{Ak�C0��                                    Bxo/?d  �          @���p�?z�H?���AN=qC)L���p�>�p�?�  A��C/ٚ                                    Bxo/N
  
�          @�ff��>\?�{A��C/�=����{?�\)A���C7�3                                    Bxo/\�  
�          @�\)��33>��R@�RA�=qC0c���33�E�@Q�AΣ�C=�                                    Bxo/kV  E          @����z�>�p�@P  B�HC/0���z῅�@G
=B
G�CAL�                                    Bxo/y�  
�          @��R���?u@QG�B�RC'8R����   @W�B�C:                                    Bxo/��  w          @����z�\@N{B�C9@ ��z�ٙ�@0��A��CJO\                                    Bxo/�H  �          @���������@(�A��C5�q������?�
=A��C?��                                    Bxo/��  T          @�p����>u@)��A��C1G�����fff@ ��AҸRC>0�                                    Bxo/��  
�          @�p���33?��@��A�(�C-=q��33��ff@�A��C9
                                    Bxo/�:  
�          @�
=��p�?E�@�
A�Q�C+z���p����@�HA�=qC6޸                                    Bxo/��  �          @�
=��Q�?W
=@�A��C*���Q��G�@\)A���C5(�                                    Bxo/��  
�          @�\)���
?Tz�@��AƸRC*Ǯ���
�u@!�A�(�C6��                                    Bxo/�,  
�          @�  ����?fff@'�AٮC)�\���þ�=q@0��A�{C7�                                    Bxo/��  
�          @�����\)?aG�@2�\A�33C)�R��\)��Q�@9��A�C8�                                    Bxo0x  
�          @�Q���Q�?�  @E�B�C(&f��Q쾽p�@N{B  C8p�                                    Bxo0  
�          @�Q���33?��@(�A�=qC'���33�#�
@*�HA��HC4+�                                    Bxo0)�  �          @�Q����H?�=q@p�A�{C(  ���H��@+�A�Q�C4T{                                    Bxo08j  �          @�Q����
?�{@�A�z�C'���
=#�
@'
=A�ffC3��                                    Bxo0G  �          @�����=q?fff@#�
A�\)C)����=q�k�@,��A�ffC6��                                    Bxo0U�  "          @�\)��=q?h��@<��A�=qC)E��=q�\@Dz�B�C8�                                    Bxo0d\  �          @�
=����?��R@7
=A�G�C%ff���׽#�
@G�B��C4��                                    Bxo0s  
�          @�
=����?L��@AG�A�\)C*}q���׿�\@E�B�C:�                                    Bxo0��  
�          @����{?��
@G
=B�HC'����{��{@P��B  C8!H                                    Bxo0�N  �          @�  ��
=?��@Y��B��C&����
=��
=@b�\B��C9h�                                    Bxo0��  
�          @�����z�?h��@S33B��C(���z��@XQ�B�RC:\)                                    Bxo0��  T          @�G���p�?�\)@dz�B�C�=��p�>��@���B/�RC0s3                                    Bxo0�@  
�          @��R����?�(�@hQ�B(�C�����=�Q�@���B3  C2�3                                    Bxo0��  
�          @���Q�?��@p  B&�\C!���Q쾨��@}p�B2�C8��                                    Bxo0ٌ  �          @�ff�mp�?�{@xQ�B,z�CQ��mp�>�@���BE
=C2�                                    Bxo0�2  �          @���e�@ ��@x��B-�
C�3�e�>��@��
BJp�C/��                                    Bxo0��  �          @����z�H@
�H@\��BQ�C��z�H?
=@���B5�
C+s3                                    Bxo1~  �          @�p��tz�@  @aG�B
=C}q�tz�?#�
@�(�B;�\C*�=                                    Bxo1$  T          @�p��n{@$z�@Z�HBp�CW
�n{?z�H@�p�B=33C%L�                                    Bxo1"�  �          @�p��l��@{@`  B�CB��l��?\(�@�ffB?ffC&��                                    Bxo11p  �          @�ff�j=q@�\@n{B#(�C���j=q?��@�=qBE�C*��                                    Bxo1@  "          @�
=�dz�@-p�@b�\B=qC���dz�?���@��\BE(�C#h�                                    Bxo1N�  
�          @����33@{@J=qB�C�H��33?E�@q�B(C)k�                                    Bxo1]b  "          @�(��r�\@#�
@Q�B�C�q�r�\?��@���B6C$�R                                    Bxo1l  
�          @�z���33?}p�@[�B\)C'33��33��@b�\B
=C:�                                    Bxo1z�  "          @�������@33@1�A�z�C�3����?G�@W�B�C)�R                                    Bxo1�T  
�          @�ff��(�@!�@%A�
=C���(�?�ff@XQ�BG�C#�                                     Bxo1��  �          @��R����@{@p�A�Ch�����?��@N�RB	�C$
                                    Bxo1��  
�          @�\)��z�@��@{A��C5���z�?�\)@@  A��\C#��                                    Bxo1�F  "          @�\)����?�Q�?�
=A��C�{����?��\@"�\A��HC(xR                                    Bxo1��  
�          @�  ��
=@��?�\)A���C8R��
=?���@(�Aȏ\C"�                                    Bxo1Ғ  �          @�  ����@�R?��HAB=qC�q����?�@A���C @                                     Bxo1�8  
�          @�����H@@  ?8Q�@�\)C@ ���H@{?�A��
C�R                                    Bxo1��  �          @������@@  ?=p�@��C^����@p�?�{A��RC#�                                    Bxo1��  �          @�  ���\@B�\>�@�=qC�����\@'
=?У�A��RC��                                    Bxo2*  T          @��R�|(�@e������\)C	���|(�@{��u�#�
C\                                    Bxo2�  "          @����z�@N�R�����C���z�@K�?J=qA	G�C��                                    Bxo2*v  �          @��R��=q@!G�@�A��RC���=q?�
=@E�B
=C"�)                                    Bxo29  �          @����@Q�@   A��\C���?�@1G�A�  C#+�                                    Bxo2G�  �          @�{��33@,(�@�\A���CO\��33?˅@I��B�HC�R                                    Bxo2Vh  �          @�p����@a�?s33A=qC�����@8��@\)A��C.                                    Bxo2e  �          @�p����@u�?
=q@�\)C�����@U�?�p�A�\)C�f                                    Bxo2s�  T          @�33�a�@x�þ�=q�<��C=q�a�@l(�?�  A\��C�q                                    Bxo2�Z  �          @�{���?�@Mp�B	C-������B�\@J=qBffC=B�                                    Bxo2�   
�          @��R���
>�Q�@S�
B�C/�
���
�s33@L(�BQ�C?�
                                    Bxo2��  
�          @���=q?�Q�@+�A���C&)��=q=�\)@;�A���C30�                                    Bxo2�L  
�          @��H�h��@,(�@@��B�C���h��?�=q@s�
B233C�3                                    Bxo2��  "          @�(����H@S�
?���AX��C�3���H@#33@�RA�  C�{                                    Bxo2˘  �          @�z��|(�@r�\?n{AffC!H�|(�@I��@�
A���Ck�                                    Bxo2�>  T          @�p��XQ�@�{���
�E�B�� �XQ�@�(�?У�A�CB�                                    Bxo2��  �          @�ff�b�\@�p�>8Q�?��
C ���b�\@���?���A�ffCc�                                    Bxo2��  
�          @�{�a�@��?Q�A��C �R�a�@n{@�HA�ffC}q                                    Bxo30  �          @�33��33@\��?��AU�C����33@,(�@!�A�z�C@                                     Bxo3�  
�          @�33��=q@n{?��
A#�
CG���=q@C33@
=A��RC�=                                    Bxo3#|  "          @��H��(�@g
=?�p�AD(�Cz���(�@8Q�@ ��A��HC��                                    Bxo32"  "          @����=q@^�R?�\)A�G�C#���=q@ ��@C33A�33C��                                    Bxo3@�  �          @��
��\)@Tz�?�\A�
=C}q��\)@��@9��A�33C�\                                    Bxo3On  "          @�G����@X��?�p�Av{CǮ���@%@)��A�33C�                                    Bxo3^  �          @�z���\)@c�
?��RA���C���\)@"�\@K�B\)C�                                    Bxo3l�  �          @�ff��  @`��?�z�A�\)C�q��  @(��@6ffA�C�H                                    Bxo3{`  "          @�����33@fff?�Q�A^{C���33@333@,(�A�Q�C��                                    Bxo3�  "          @�Q���ff@XQ�?��An�HCL���ff@#33@,��A��Cz�                                    Bxo3��  �          @������@tz�?Tz�A   C�=����@N�R@��A���Cn                                    Bxo3�R  
�          @�Q�����@mp�?+�@�{Ch�����@L(�@ ��A��RC�=                                    Bxo3��  �          @��R��(�@dz�?���A��\C�\��(�@(��@@��A�=qC�                                    Bxo3Ğ  "          @�������@U�?У�A~�RCaH����@�R@0  Aܣ�C޸                                    Bxo3�D  �          @�G����
@E@A��RC.���
@   @Tz�B�C�=                                    Bxo3��  �          @�=q��=q@n�R?z�HA
=C����=q@Fff@�\A�Q�C�)                                    Bxo3�  "          @�33����@�G�?��ATz�C+�����@^{@8Q�A�\)C5�                                    Bxo3�6  T          @�(���{@{�?�Q�AYC
����{@G�@1�A�33C�                                    Bxo4�  �          @ə����@�p�?xQ�A�\C�����@p  @"�\A�=qCO\                                    Bxo4�  
�          @�=q��p�@�\)>�=q@{C����p�@�=q?�33A�z�C	c�                                    Bxo4+(            @�\)����@�Q�.{���
Cu�����@�Q�?�  A_�C�                                    Bxo49�  b          @������\@�
=?\(�A ��C� ���\@g
=@ffA��\C.                                    Bxo4Ht  �          @�33����@�=q=#�
>���C� ����@�Q�?�\)Aw33C�                                    Bxo4W  �          @ƸR����@��
�O\)���C@ ����@�33?n{A	�CaH                                    Bxo4e�  "          @��
���\@�ff�k��
ffC}q���\@�\)?E�@�
=CO\                                    Bxo4tf  �          @��H�hQ�@�
=��G��?
=B�\�hQ�@��?��@���B��                                    Bxo4�  T          @���aG�@�33��  ���RB��aG�@�p�=#�
>�p�B�\                                    Bxo4��  
�          @����s�
@��
���H��33C��s�
@�{�#�
�L��C)                                    Bxo4�X  
�          @�{�E�@�(��I����HB��ÿE�@�{��z��7\)B�p�                                    Bxo4��  T          @�p�����@���HQ�� =qBң׿���@��
���8(�B�k�                                    Bxo4��  
�          @�p��.{@�p��`�����BĀ �.{@��
�˅�z�HB��q                                    Bxo4�J  
�          @�p���ff@����r�\�G�B��f��ff@�33�   ��B�W
                                    Bxo4��  �          @��
��(�@����mp��p�B�\��(�@�=q�����B���                                    Bxo4�  T          @��
��Q�@�\)�mp���\Bߊ=��Q�@��ÿ�Q���z�B׀                                     Bxo4�<  
�          @��H�Y��@���qG��!Q�B�.�Y��@�p����H��33B��                                    Bxo5�  	�          @�녿(�@�����\�3=qB�33�(�@�  �
=��=qB�W
                                    Bxo5�  �          @�����=q@u���G��133B���=q@����=q�ĸRB��H                                    Bxo5$.  
�          @�\)��=q?�G��e�33C����=q@:�H�.�R�ܸRC�R                                    Bxo52�  	�          @�ff��p�?L����(��0�
C).��p�@p��dz����C
=                                    Bxo5Az  �          @�p����?��H�u�"�C$G����@   �J=q��\C�f                                    Bxo5P   
(          @�z��qG�@��u��&�\C��qG�@S33�5��ffC
                                    Bxo5^�  
�          @����U�?�G���ff�J��C�H�U�@>�R�hQ��p�C
�                                    Bxo5ml  "          @����n�R?�Q���(��7�
C�3�n�R@333�W
=�
=C
                                    Bxo5|  
�          @�ff�c�
?������<
=C���c�
@:=q�Vff�ffC�                                    Bxo5��  
�          @����L��?W
=����V=qC%G��L��@��y���1�RC��                                    Bxo5�^  T          @��
���R�����\8RCE�{���R?����p�Cc�                                    Bxo5�  �          @��\�1G�@%��s33�4�C	��1G�@p  �*=q��B���                                    Bxo5��  T          @�(��_\)@��Ϳz���(�C��_\)@�33?h��A��CxR                                    Bxo5�P  
�          @�z��o\)@p  �޸R��33C���o\)@��
���
�P��C33                                    Bxo5��  b          @��\�]p�@I���,(���G�C	��]p�@x�ÿ��t��C��                                    Bxo5�  T          @����l��@S33��ff��{C
@ �l��@n{�
=q��p�C�\                                    Bxo5�B  
�          @�=q�n�R@q녾�(�����C���n�R@l��?h��A��C@                                     Bxo5��  T          @�z��
�H@g
=�k��$��B����
�H@�p������p�B��H                                    Bxo6�  T          @��R�(�@hQ�����Gp�B��ÿ(�@���5���z�B�\                                    Bxo64  T          @�����@��
�j�H�#  B�𤿇�@�(��G���ffB�ff                                    Bxo6+�  T          @���
=@��Ϳ�����p�B�8R�
=@��
>�?�G�B�                                    Bxo6:�  �          @���qG�@s�
?�33A���C���qG�@@��@7
=A�p�Cn                                    Bxo6I&  "          @����{@L��?�Q�AlQ�C5���{@ ��@��A͙�C��                                    Bxo6W�  �          @���Vff@��\?���A]B�p��Vff@fff@0  A��C�f                                    Bxo6fr  T          @��
�n�R@{�?�(�Ar�RC�\�n�R@L(�@.�RA�Q�C��                                    Bxo6u  T          @���]p�@�  ?�{A��HC޸�]p�@HQ�@G�B�C	�                                    Bxo6��  
�          @�(��q�@c�
@
=A���C�3�q�@(Q�@L��B33C.                                    Bxo6�d  
�          @�����@K�?p��A��C=q���@*�H?��HA���C�{                                    Bxo6�
  �          @����Q�@dz�W
=�	��C{��Q�@h��>�
=@��C�                                     Bxo6��  
Z          @�p��P��@�G������HC ���P��@�  �
=��G�B��q                                    Bxo6�V  
Z          @�����ff@��>{���HC����ff@I���G�����C+�                                    Bxo6��  �          @��
��(�?�G��^{�
=C�R��(�@%�0����C�                                    Bxo6ۢ  �          @�33�j�H@��\���ffC)�j�H@S�
��R���HC	�                                    Bxo6�H  	�          @�G����
@�33��\)��\)B�=q���
@�33>B�\?���B���                                    Bxo6��  
�          @��H���H@�����p����RB����H@�p��#�
�ٙ�B�=q                                    Bxo7�  
�          @��H����@�{�8Q����HB۽q����@��Ϳ�(��L  B֞�                                    Bxo7:  x          @����@Q��l���*G�B��f�@��\�����
=B�                                    Bxo7$�  /          @���@)���}p��@�C޸��@tz��6ff��\B�                                    Bxo73�  �          @�p�����@~�R�N{�\)B�����@��\�ٙ���  Bب�                                    Bxo7B,  �          @�ff�aG�@��ÿ�ff�2=qC&f�aG�@���>�33@j�HCO\                                    Bxo7P�  
�          @��\�b�\@�ff>#�
?�C#��b�\@z=q?�ffA���C.                                    Bxo7_x  �          @��H�s�
@���?(��@�=qC}q�s�
@dz�?��HA�ffC޸                                    Bxo7n  
�          @��H�j=q@����R�N�RCB��j=q@�G�?��A4Q�C+�                                    Bxo7|�  	�          @�Q��i��@��ÿ���p�C33�i��@~{?Y��AffC��                                    Bxo7�j  "          @�  �z�H@s33���
�L��C�{�z�H@g�?�
=AHQ�C	G�                                    Bxo7�  
�          @�\)�A�@��ÿ��H�x��B����A�@���<�>���B���                                    Bxo7��  T          @�ff�'
=@�Q��  ����B���'
=@�Q�=u?+�B�=                                    Bxo7�\  �          @��
�9��@�=q������\B����9��@��R��G���B�{                                    Bxo7�  
�          @��H����@�Q�@5�B�
B�k�����@7
=@\)BQ�B�
=                                    Bxo7Ԩ  T          @�{����@,(�@�33BY(�B�����?���@�Q�B��C�                                    Bxo7�N  �          @����
=@J=q@eB)�B��=�
=?���@�p�B`(�Cs3                                    Bxo7��  T          @�ff�N{@i��?�{A�p�Cz��N{@5@<��B	��C
�)                                    Bxo8 �  
�          @��R�P��@(��@A�B\)C���P��?\@n{B833C
=                                    Bxo8@            @��Ϳ�ff@g
=@EB�B���ff@=q@��HBV�B�aH                                    Bxo8�  
g          @�
=���@��
@(��A�p�B��
���@Q�@z=qBB�B�L�                                    Bxo8,�  
�          @����(�@���@{A�{B��Ϳ�(�@O\)@n{B5�RB��
                                    Bxo8;2  
          @����"�\@�  @33A�G�B��"�\@C33@^{B$�
C��                                    Bxo8I�  T          @����@��H?�A��B�q�@aG�@I��B��B�ff                                    Bxo8X~  /          @�=q�(�@�z�?�ffAc\)B�k��(�@~�R@.{A�B�{                                    Bxo8g$  "          @�z�Ǯ@��?s33A#
=B�aH�Ǯ@�\)@!�A�{B�z�                                    Bxo8u�  "          @�z῏\)@�{?W
=AQ�B�LͿ�\)@�z�@p�A�{B�\                                    Bxo8�p  
�          @��\�B�\@�z�?�\)AB�\B��)�B�\@�Q�@-p�A���B�.                                    Bxo8�  �          @��H�n{@�\)>�{@g
=B�33�n{@�33?��RA��Bɽq                                    Bxo8��  T          @��
���
@�  ���Ϳ�\)B�G����
@�  ?���A��HB�Q�                                    Bxo8�b  �          @�(��n{@��þL����B�
=�n{@�=q?��RA��RB��
                                    Bxo8�  
�          @���?!G�@�(��
=��G�B�.?!G�@�(��0�����B��=                                    Bxo8ͮ  T          @��?�=q@�z��	����{B�.?�=q@�=q���H���\B��                                    Bxo8�T  �          @�ff��p�?B�\@FffB
=C*��p����@K�B	  C75�                                    Bxo8��  "          @��\��
=?�33@2�\A�C&�R��
=>W
=@@��A��C1��                                    Bxo8��  �          @��\���
@G�@p�A�  C�����
?��@>{A��C&ٚ                                    Bxo9F  �          @�Q����@{?���A�\)C���?��
@   A�z�C�                                    Bxo9�  T          @�ff�<(�@�p�?(�@ۅB�G��<(�@qG�?�33A�
=B��)                                    Bxo9%�  �          @������@�p���(���  B�aH���@�=q?��A<��B�\)                                    Bxo948  
          @�z��\)@�녿�����B׀ ��\)@��R?��A=p�B��                                    Bxo9B�  
�          @�
=����@�{�L�Ϳ
=B�zῨ��@�{?���A�z�B���                                    Bxo9Q�  �          @�Q��<��@h����H��z�C ���<��@�
=��z��K�B��                                    Bxo9`*  T          @���
@��@Q�A�G�B�����
@j�H@\��B��B�z�                                    Bxo9n�  T          @�(��  @�33@A�B�
=�  @n{@Z�HB�RB�33                                    Bxo9}v  "          @����p  @s33?�A��C���p  @C33@9��A�G�C�                                    Bxo9�  
Z          @����Tz�@vff�E��
=qC���Tz�@y��>�@��Cn                                    Bxo9��  G          @�
=�J=q@��Ϳ�z��H  B��=�J=q@��>8Q�?�(�B��=                                    Bxo9�h  
�          @����E�@���>�p�@��
B�k��E�@o\)?���A�=qCp�                                    Bxo9�  "          @�  ��{@�Q������B���{@�{��R��  B�\)                                    Bxo9ƴ  
�          @�ff��G�@��R�[���
B�p���G�@�녿�(���G�B�33                                    Bxo9�Z  	�          @�{�Dz�@�\)�
=q���B���Dz�@�{�@  ��p�B�k�                                    Bxo9�   "          @��R�J=q@�\)�����EB�u��J=q@�(�>k�@ffB���                                    Bxo9�            @������@'
=@z�A��C������?���@1G�A��
CQ�                                    Bxo:L  
5          @�z����\@5?�\A�{C�\���\@
=q@#33A��
C�H                                    Bxo:�  T          @�p���33?��R@
�HA��C����33?�p�@+�A�{C&k�                                    Bxo:�  
�          @����33@��@��A�\)C�\��33?��@@��A���C"\)                                    Bxo:->  
�          @�������@-p�@Q�A�33C�3����?�{@FffA�  C+�                                    Bxo:;�  �          @�ff���
@%�?��
Ao�C8R���
@   @\)A�C�3                                    Bxo:J�  �          @������@;�?�G�A\)Cn���@�R?���A��CO\                                    Bxo:Y0  
�          @�(����@%?��HA�G�C�����?���@=qA�  C�
                                    Bxo:g�  �          @����@�\?�p�A���C����?��@ ��A���C%u�                                    Bxo:v|  "          @�z���=q@�?�{A��CO\��=q?��H@=qȀ\C#0�                                    Bxo:�"  �          @�(����?���@(�A��HC�H���?���@*�HA�G�C%�                                    Bxo:��  T          @�ff��ff?Ǯ@2�\A�
=C!�)��ff?&ff@HQ�B��C,
                                    Bxo:�n  
�          @�z����?
=q@�\A��C-�3�������@
=A�Q�C5�                                    Bxo:�  �          @��H��  =�\)@(��A�  C3)��  �8Q�@"�\Aޏ\C<�
                                    Bxo:��  �          @��R��p�@�>W
=@�C\��p�@��?h��A�C�                                     Bxo:�`  �          @�{��p�?޸R?��HAo�C!aH��p�?�p�?�z�A�=qC&�f                                    Bxo:�  �          @�����p�@   ?+�@���C�
��p�?��H?��RAL��C!��                                    Bxo:�  
�          @�  ����?��R>�=q@5�C$W
����?��?333@��C%�{                                    Bxo:�R  "          @�(���=q@��?�G�A%��C���=q?޸R?˅A��\C!                                      Bxo;�  �          @�{���R?��?�33A>{C"����R?�p�?�=qA��
C&�                                     Bxo;�  
�          @����G�?�(�?�(�AA�C'�
��G�?L��?\As�C+�q                                    Bxo;&D  "          @������R?�p�?��AX  C'Q����R?G�?�33A���C+�f                                    Bxo;4�  
�          @����-p����R?�33A�p�CL���-p��\?=p�AU�CQL�                                    Bxo;C�  
Z          @��
=�\)����W
=�C�h�=�\)��ff��  ���C�o\                                    Bxo;R6  "          @�=q�*=q�?\)@(Q�B(�Cdc��*=q�hQ�?У�A���Ci��                                    Bxo;`�  "          @����{�&ff>��R@c�
CS�=��{�$z�����{CS��                                    Bxo;o�  "          @��R��G���  ?uA,��C6�
��G���?Y��A  C9�f                                    Bxo;~(  �          @��\���Ϳ�G���p��m�CF����Ϳ�=q��\)��33CA�
                                    Bxo;��  
Z          @�z��j=q�A������ffC[�{�j=q����?\)�  CS�H                                    Bxo;�t  
Z          @�G���  ?B�\�fff�ffC+����  ?}p��#�
���
C)aH                                    Bxo;�  �          @��R��(�?�p�?�Q�A��HC#+���(�?k�@33A�G�C)Q�                                    Bxo;��  T          @�����p�@ ��?�  A|z�C�=��p�?��H@
=qA���C5�                                    Bxo;�f  
�          @�������?У�?�A?�C"�=����?�(�?˅A�33C&�                                    Bxo;�  �          @�(����?�33?z�HA"�RC'޸���?Q�?��
AS�C+Q�                                    Bxo;�  
�          @�����H>��
?.{@��C0� ���H=�G�?=p�@�C2�f                                    Bxo;�X  �          @�����R�Tz�>�z�@A�C<�����R�aG�=#�
>�
=C=!H                                    Bxo<�  
�          @�����Ϳ�Q�>�(�@�ffC@ff���Ϳ�G�=�\)?:�HCA&f                                    Bxo<�  �          @�ff���
��R>�
=@�G�CL�{���
��׾k��\)CLٚ                                    Bxo<J  T          @����G��.{?@  Ap�C5�)��G��\?+�@�RC8�                                    Bxo<-�  
�          @�33��<�?�G�APz�C3��������?��HAI�C7xR                                    Bxo<<�  T          @�=q���ͼ#�
?���AW�C4����;Ǯ?��AN{C7�q                                    Bxo<K<  �          @�{���þ��?�p�AH��C6�f���ÿ��?�{A4(�C:(�                                    Bxo<Y�  
�          @�{���R���?�(�Apz�C8E���R�O\)?��AR�HC<p�                                    Bxo<h�  �          @����Q�?\)>Ǯ@�Q�C.33��Q�>�(�?�@��C/�{                                    Bxo<w.  �          @�33���\=�>���@W�C2� ���\<��
>�33@eC3                                    Bxo<��  "          @�����{=#�
?Y��A�C3����{�L��?Tz�A  C6�                                    Bxo<�z  
�          @����Q�=�G�?.{@�Q�C2����Q콸Q�?0��@�G�C4�)                                    Bxo<�   �          @�{���
>L��?E�@�C2����
��?J=q@�p�C4B�                                    Bxo<��  T          @����G�>���?��A&=qC0��G�=�Q�?���A1G�C3&f                                    Bxo<�l  
�          @�p����>�z�?�{AUp�C1��������?��AY�C5�                                    Bxo<�  �          @����  >�z�?��AQG�C1���  ����?�\)AUC4�R                                    Bxo<ݸ  "          @�z���Q�>��?�33A4  C/�H��Q�=�\)?�(�A?
=C3B�                                    Bxo<�^  �          @�����=q?�\?O\)@�{C.����=q>��?n{Az�C1k�                                    Bxo<�  �          @�z���G�>Ǯ?��\A�C0���G�=�Q�?��A*�\C3�                                    Bxo=	�  a          @����
=>��?�p�AA��C1\)��
=��Q�?�  AE��C4�f                                    Bxo=P  �          @�z���Q�>���?�33A4��C0����Q�#�
?���A<(�C4\                                    Bxo=&�  y          @�(���\)>�?���A<  C/0���\)>�?��
AI�C2�3                                    Bxo=5�  
�          @��
��z�?E�?��A[\)C,E��z�>�33?�ffAt��C0xR                                    Bxo=DB  
�          @�����?J=q?�\)A}��C,
=����>��R?��
A�\)C0�H                                    Bxo=R�  
�          @�G���
=?aG�?�
=A��C*�H��
=>\?�{A��
C0\                                    Bxo=a�  
�          @������?J=q?�G�A�  C+������>�=q?�z�A�Q�C18R                                    Bxo=p4  
�          @�����
=�G�@�\A��\C2Ǯ���
��@�RA��C9��                                    Bxo=~�  a          @��
��(�<��
@ffA��HC3�\��(��\)@�A�Q�C9�q                                    Bxo=��  
�          @�ff��G�?���?��
A�p�C � ��G�?��
@{A�  C&aH                                    Bxo=�&  "          @������@U?���AX��CW
���@5�@  A�\)C�3                                    Bxo=��  
�          @����\@7
=>�p�@j�HC�H���\@)��?�33A5�C}q                                    Bxo=�r  �          @��
��ff@"�\?0��@�  C�3��ff@��?�{AW�C��                                    Bxo=�  T          @����  @(Q�>�Q�@c33CY���  @�?���A((�C!H                                    Bxo=־  T          @�ff��p�@�
=>��H@��C����p�@{�?У�A~=qC�R                                    Bxo=�d  "          @�
=�~�R@������ffC��~�R@���?�=qA'\)C޸                                    Bxo=�
  T          @�Q����ÿz�H?���A8z�C=�����ÿ��\?aG�A�
C@c�                                    Bxo>�  
�          @�G���{���?���A^=qCA����{��p�?��\A
=CD�f                                    Bxo>V  T          @\��{��
=?޸RA��C?����{��{?�{AN�RCCǮ                                    Bxo>�  	�          @�z����ÿ(��?�
=A���C:� ���ÿ�z�?�
=A~ffC?W
                                    Bxo>.�  T          @��H��(�<#�
?���Ao�
C3���(���
=?\Af�HC8�                                    Bxo>=H  T          @����ff=#�
?�  AQ�C3�{��ff�k�?xQ�AQ�C633                                    Bxo>K�  
�          @�33��{>\)?�\)AO�
C2����{�u?�{AM��C6L�                                    Bxo>Z�  �          @��\>��?B�\@�(�C0.�\>8Q�?W
=@���C2B�                                    Bxo>i:  �          @��\?8Q�?B�\@�C-G��\>�?n{A
�\C/u�                                    Bxo>w�  T          @�p���
=?fff?}p�A��C+c���
=?��?���A5�C.=q                                    Bxo>��  "          @�������?�\?�  A�C.������<��
?���A��C3�=                                    Bxo>�,  
�          @��
��Q�?E�?���A�33C,ff��Q�>u?��RA�(�C1�{                                    Bxo>��  �          @\���?.{?У�Ax��C-^����>W
=?�  A��
C1�f                                    Bxo>�x  T          @�(����
?�Q�?�\)A)�C(�����
?Y��?�33AT(�C+                                    Bxo>�  �          @�(����?��R>�G�@�p�C%�H���?��?Tz�@���C'xR                                    Bxo>��  T          @\���R?�{>�  @�C)n���R?�  ?
=q@�33C*n                                    Bxo>�j  �          @�����?���?5@��
C!�����?˅?�A7�C$                                    Bxo>�  "          @�  ��  @w��\�i��CY���  @u�?+�@�{C��                                    Bxo>��  
�          @�\)�g
=@���aG��Q�B�k��g
=@�\)>Ǯ@qG�B���                                    Bxo?
\  
�          @���-p�@��ÿE���B�Q��-p�@�G�?+�@�z�B�33                                    Bxo?  "          @������@�=q��G���RB�B����@�(�?z�@�Q�B�\                                    Bxo?'�  �          @�녿�p�@�z�J=q��  Bڙ���p�@���?=p�@�G�Bڊ=                                    Bxo?6N  �          @��=u@����G��A��B��
=u@�\)>�{@Q�B��)                                    Bxo?D�  
Z          @�
=�8Q�@�����Y�B�uÿ8Q�@�33>.{?�\)B�\                                    Bxo?S�  �          @����p�@�=q��  �iBѳ3��p�@�Q�<��
>k�BиR                                    Bxo?b@  
�          @�
=�Q�@��\��33���RBĳ3�Q�@�zᾸQ��[�B��
                                    Bxo?p�  �          @�33�z�H@��
����{Bɨ��z�H@�
=����ƸRB�G�                                    Bxo?�  �          @\���
>�@'
=A���C.޸���
�aG�@(��A�C6c�                                    Bxo?�2  �          @�  ���?�?��RAg\)C%���?z�H?���A�\)C*{                                    Bxo?��  �          @�\)��?Y��?У�A��C+
��>Ǯ?��A��
C/�)                                    Bxo?�~  
�          @�(���=q�   ?�\)A[�
C9
��=q�W
=?�Q�A>�RC<��                                    Bxo?�$  
�          @�����p�?��
>8Q�?޸RC$���p�?�
=?�@��HC%޸                                    Bxo?��  �          @������?^�R>�{@W
=C+n����?@  ?\)@�\)C,��                                    Bxo?�p  
�          @��\���?.{>�{@Y��C-G����?\)?�\@���C.ff                                    Bxo?�  T          @��\��\)?�  >���@{�C"T{��\)?�=q?W
=AC#�f                                    Bxo?��  �          @����z�@I���\)��  C^���z�@Dz�?.{@���C                                      Bxo@b  
Z          @����G�@!G�<#�
=�Q�Cff��G�@�?+�@ٙ�C8R                                    Bxo@            @��R�c�
@XQ��*=q��C���c�
@|�Ϳ�Q���
=C                                    Bxo@ �  `          @�ff���@[���p���Q�C}q���@qG��G�����C	޸                                    Bxo@/T  �          @�
=�s33@r�\��(���G�C��s33@��k���
CT{                                    Bxo@=�  
Z          @�����
=@aG���\��33C#���
=@w
=�L�����RC	�                                    Bxo@L�  T          @���z=q@dz������HC	���z=q@��\���ap�C��                                    Bxo@[F  
(          @����z�@_\)��\���C����z�@y�������)C�                                    Bxo@i�  "          @��\�z�H@&ff�Z=q��Cn�z�H@Y���'���ffC)                                    Bxo@x�  �          @������@W
=�#�
���Cu����@y�������~ffC(�                                    Bxo@�8  
�          @�=q��(�@9���{�ɅC����(�@\(������33C޸                                    Bxo@��  �          @�33��Q�@<(������  C޸��Q�@Z�H���a��CǮ                                    Bxo@��  
�          @�33���\@0�������\C����\@QG�����up�C}q                                    Bxo@�*  	�          @�����@U�=q����C�3����@vff��(��e�C	�R                                    Bxo@��  �          @�\)�h��@j=q�>�R��C�{�h��@�녿�Q�����C8R                                    Bxo@�v  
�          @����Vff@����;���(�CǮ�Vff@�z��ff���
B���                                    Bxo@�  
�          @��H�mp�@����
�H���C�H�mp�@�ff��  ��C J=                                    Bxo@��  T          @��
�j=q@����H����CǮ�j=q@�
=��  �=��B��\                                    Bxo@�h  �          @\�}p�@p  �*=q���C���}p�@�����{�vffC��                                    BxoA  "          @��H�u�@x���,(���C���u�@�{��{�t��C�=                                    BxoA�  
�          @����mp�@tz��9����\C��mp�@�{��=q��=qC�)                                    BxoA(Z  
�          @��H���\@n{�%�����C	�����\@�Q��ff�k�C                                    BxoA7   �          @��H�`  @qG��QG����C�{�`  @�\)�������B���                                    BxoAE�  �          @��
�j=q@q��Fff��  C)�j=q@�{�����C�                                     BxoATL  �          @�=q�P  @�G��=p����
C ޸�P  @������\)B��H                                    BxoAb�  �          @��H�H��@�  �:=q��Q�B���H��@�33��p���G�B��H                                    BxoAq�  
�          @����\(�@u�1G���(�C�f�\(�@�����H����B��f                                    BxoA�>  "          @��H��Q�@}p��  ��(�CO\��Q�@��Ϳ��2{CJ=                                    BxoA��  �          @�(���ff@s33�p���G�C)��ff@z=q=L��?   CG�                                    BxoA��  "          @�z����
@8Q�0���ϮC�\���
@=p�=u?��C#�                                    BxoA�0  �          @�33��(�@  ����z�RC\)��(�@%��G��=qCW
                                    BxoA��  
Z          @�����\)?��Ϳ�33��33C#�R��\)@녿�
=�\Q�C��                                    BxoA�|  T          @�ff��33@)����
��C� ��33@E�����H(�C�=                                    BxoA�"  
(          @�ff����@%�\)����C������@H�ÿ�G���z�C�q                                    BxoA��  �          @�{��\)@W
=��\)�L��C�q��\)@fff������C+�                                    BxoA�n  
�          @����@S�
�
=q����CO\���@U>��R@:=qC{                                    BxoB  �          @��
��  @!G���
=�\)Ch���  @"�\>aG�@Q�C0�                                    BxoB�  �          @�����Q�@(Q쿌���*=qCz���Q�@4z����z=qC��                                    BxoB!`  �          @�����33@@���Fff���HC����33@l(��\)��
=C	�q                                    BxoB0  T          @�ff�K�@,�������/��C���K�@g��Mp����CJ=                                    BxoB>�  
�          @�=q��  ?�׼��
�#�
C!�)��  ?�=q>�@�=qC"^�                                    BxoBMR  
�          @��H��  ?�\)�����C!�q��  ?�����Q�O\)C!:�                                    BxoB[�  "          @������R@{�����7�Cٚ���R@�Ϳ�����C��                                    BxoBj�  
�          @��H���@E��s33�(�C0����@N{�����
=C\                                    BxoByD  �          @��
��@)���Q���ffC���@1G�������C�q                                    BxoB��  T          @�p���p�@'���{�K�C33��p�@7��(������C
=                                    BxoB��  "          @�ff��
=@(����
�eG�C���
=@0  �^�R�CT{                                    BxoB�6  
�          @�Q���
=@>{����t��Cff��
=@Q녿Y�����HCٚ                                    BxoB��  	�          @ə����@'��У��pQ�Cn���@<�Ϳn{�33C�q                                    BxoB  	�          @��H����@5������C0�����@S�
��  �\Q�CB�                                    BxoB�(  
f          @����p�@G
=�333��Q�CY���p�@mp���Q���{C�=                                    BxoB��  
�          @��H���?����,(����HC(�)���?�p�����ffC"��                                    BxoB�t  
�          @�����
�k����
=C6:����
>������Q�C1�                                    BxoB�  �          @�G���ff�\�����C7�3��ff>L����
��\)C1�)                                    BxoC�  �          @�����
=>B�\������C2#���
=?5� ����C,�3                                    BxoCf  �          @�ff��\)��\)�333��ffC4� ��\)?!G��.�R�ҏ\C-}q                                    BxoC)  �          @�Q���ff��G��{��
=C8u���ff>8Q��   ���C2(�                                    BxoC7�  T          @�\)��33�!G��(Q���  C:����33=#�
�,����Q�C3��                                    BxoCFX  z          @�
=���\���
�&ff�ǅC4ٚ���\?\)�"�\�\C.=q                                    BxoCT�  .          @ȣ����\>�ff������HC/�=���\?xQ���H����C*��                                    BxoCc�  �          @�����\?J=q�����{C,@ ���\?����
=����C'L�                                    BxoCrJ  "          @�G���  ?L������C,)��  ?�{�ff���\C&�3                                    BxoC��  
�          @�=q��  ?�����?\)C&�f��  ?ٙ��h���G�C$&f                                    BxoC��  "          @�33��@	�����R�333C ���@
=q>k�@
=C�R                                    BxoC�<  "          @��
��@�Ϳ�����RC����@G�<#�
=�G�C�                                    BxoC��  �          @ʏ\���?�\���\�(�C#� ���?�(������
=C!�=                                    BxoC��  T          @˅��{@+�?:�H@�z�C����{@=q?�\)AH  C�                                    BxoC�.  "          @�(���\)@+�?+�@���C�R��\)@�?��A=��C�                                    BxoC��  �          @�(���=q@!G�?�@���C����=q@�
?���A$(�CY�                                    BxoC�z  �          @�Q���ff@�H?&ff@���C���ff@(�?�p�A6�\C��                                    BxoC�   "          @�����
=@J�H����(z�C#���
=@U��u��HC��                                    BxoD�  T          @����-p�@����:�H�B��R�-p�@��
���
��\)B�B�                                    BxoDl  T          @���8Q�@���"�\��Q�B��f�8Q�@�33��=q�L(�B��f                                    BxoD"  
Z          @��H�8��@�(��,����(�B����8��@��Ϳ�(��`  B                                    BxoD0�  �          @��H��@�  �����\B��f�H��@�{��\)�((�B�\)                                    BxoD?^  �          @�=q�G
=@�  �	����  B��=�G
=@�(��k��
=B�                                    BxoDN  �          @����Vff@���0  ����C}q�Vff@�������\B��                                    BxoD\�  �          @��\�E�@n�R�G
=��C���E�@�(��ff��33B�G�                                    BxoDkP  �          @��
�l(�@}p����
=C���l(�@������'�CE                                    BxoDy�  
�          @������@_\)�z�H�C����@hQ콸Q�aG�C                                      BxoD��  �          @�33��  @~{�s33�z�C
����  @��H=L��>��C	Ǯ                                    BxoD�B  �          @�Q����@q녿����D  CaH���@\)��33�N{C�H                                    BxoD��  
�          @�����{@\)��G��\Q�C����{@�������
C	�
                                    BxoD��  �          @�Q���33@��׿��Q�C
����33@�  �Ǯ�c33C	@                                     BxoD�4  T          @�������@~{��G��2ffC������@��;�  �(�Cc�                                    BxoD��  
Z          @ə����\@��R��Q��yp�CxR���\@���
=q���RC�                                     BxoD��  �          @�����Q�@�z�aG����RC�\��Q�@��R>�\)@#33C)                                    BxoD�&  
�          @˅�q�@��
��
=�P��B��q�q�@�=q�B�\��G�B�z�                                    BxoD��  
�          @�\)�=q@�33��Q���(�B�z��=q@��������B�{                                    BxoEr  �          @�G��I��@����
�E��B�(��I��@��H������B�z�                                    BxoE  
�          @����S33@��n{��RB���S33@�Q�>���@:=qB��3                                    BxoE)�  
(          @�  �1G�@�(������B�Ǯ�1G�@�Q�Y����(�B�p�                                    BxoE8d  T          @�z��-p�@�  �
=q����B����-p�@��
�aG��z�B�                                    BxoEG
  T          @����@�(�����\B��)��@�  �Tz���
B�                                    BxoEU�  �          @���E@�  ���H���\B�{�E@��ÿ   ���HB�(�                                    BxoEdV  T          @��R��@����/\)���B�8R��@�G����R�jffB�                                      BxoEr�  
�          @�Q�� ��@�z������{B�� ��@�Q�aG����B��)                                    BxoE��  
�          @���U@�z��33���B�p��U@�Q�h���
=B�\                                    BxoE�H  �          @�z��L(�@�  �����(�B����L(�@��R����K�B�G�                                    BxoE��  T          @��R�K�@����
��=qB����K�@������0��B�                                    BxoE��  "          @\�fff@����(Q��̏\C�3�fff@�����
�ip�B�L�                                    BxoE�:  �          @�\)�^�R@��@����C�=�^�R@�G���33��{B�
=                                    BxoE��  �          @��<(�@���8����(�B���<(�@��
��
=�~�RB�3                                    BxoEن  �          @�{���R@����b�\�\)B� ���R@�Q������
B݅                                    BxoE�,  
�          @���j=q@���=q���\C+��j=q@�{���R�6{B�                                    BxoE��  
�          @�  �mp�@���H����p�C�3�mp�@�{���R����B��3                                    BxoFx  !          @�G��e@�  �[����C&f�e@��R����B��R                                    BxoF  �          @ҏ\�n�R@����U���\)C#��n�R@�ff����G�B��                                    BxoF"�  
�          @��H�hQ�@���ff��z�B��hQ�@��\��=q�  B��                                    BxoF1j  �          @ҏ\�mp�@���AG���\)C���mp�@�33��=q���B�{                                    BxoF@  
�          @ҏ\�N{@l(�����%�\C)�N{@�(��Q���B���                                    BxoFN�  T          @�=q��Q�@�=q�"�\���C����Q�@�녿�(��Y�CG�                                    BxoF]\  "          @�
=���@   �Y�����C�{���@P  �,(���\)C:�                                    BxoFl  �          @�z���\)@U��{�E��Cff��\)@e��   ��Q�C��                                    BxoFz�  �          @�=q��Q�@Vff�xQ���C�\��Q�@^�R�����C�                                    BxoF�N  
�          @�ff��(�@s�
=��
?5Ch���(�@j�H?�G�AG�Cc�                                    BxoF��  
3          @�  ��Q�@��\?\)@��RC޸��Q�@tz�?��A[�
C��                                    BxoF��  T          @�ff���
@:�H?c�
@��RC�����
@'�?���Ac�
C
=                                    BxoF�@  �          @����{@.{?��\A�\Cu���{@��?�33Ao�C=q                                    BxoF��  �          @�p����@Q�?�A)G�C�{���@8��?�Q�A�G�C�                                    BxoFҌ  
�          @�z����H@HQ�?�ffAa��C�����H@*=q@G�A���C�=                                   BxoF�2  
�          @�\)��ff@QG�?�(�A-p�C���ff@8Q�?�p�A��\C.                                   BxoF��  "          @�33����@Q�?���A.�HCB�����@8Q�?�(�A��\Cp�                                    BxoF�~  �          @�����@A�?�G�A8z�C� ���@(Q�?��HA�p�C�
                                    BxoG$  "          @����p�@9��?�33AMG�Cٚ��p�@{@z�A�33C��                                    BxoG�  T          @�z����R@:�H?��RAX��C�)���R@p�@
=qA��RC��                                    BxoG*p  "          @�33��\)@4z�?�=qAB�\CǮ��\)@=q?�p�A�=qCG�                                    BxoG9  T          @������@]p�@p�A���C{����@3�
@?\)A�33C�                                    BxoGG�  "          @��H��33@W
=@0  A�  C�H��33@%�@^�RB�
C�3                                    BxoGVb  
�          @��H��z�@>{@+�A��HC����z�@{@U�A�=qC�=                                    BxoGe  
�          @ə���ff@/\)@-p�Ạ�C���ff?��R@S33A��C
                                    BxoGs�  T          @�����(�@�R@'�A�C@ ��(�?�G�@I��A�33C!�                                    BxoG�T  "          @�����@�
@,��A�\)C�R��?�=q@K�A�\)C#\                                    BxoG��  
�          @�=q����@=q?�A���Cs3����?��@Q�A���C!+�                                    BxoG��  
�          @�(���z�@(Q�?�G�A6�RC���z�@\)?�\)A���CW
                                    BxoG�F  �          @��
��  @?�z�AL��C�)��  ?�?�Q�A�(�C!�                                    BxoG��  
�          @��
����@#�
?�A�ffC(�����@�\@Q�A�=qCǮ                                    BxoG˒  
�          @��
��\)@�?�Q�AR{C���\)?�33?�(�A��\C!��                                    BxoG�8  
�          @�33��@'
=?�z�A(��CL���@\)?�\A�\)Cn                                    BxoG��  "          @��H��
=@7�?�\)AH��C^���
=@(�@�\A�(�C��                                    BxoG��  �          @�����33@:=q?��
Ab{Cs3��33@��@��A�(�Cp�                                    BxoH*  T          @Ǯ����@;�?��\A�C������@%?��HA�Cn                                    BxoH�  
�          @�G����@N{?Tz�@���C���@:=q?���Al��C8R                                    BxoH#v  �          @˅��
=@:=q?��A'33C���
=@"�\?���A���C�                                    BxoH2  
�          @�G�����@�R@   A�
=CL�����?�z�@#33A�=qC }q                                    BxoH@�  
�          @�������@(��?��HA}p�C�����@��@�
A��C^�                                    BxoHOh  T          @�{���@qG�?�  A:�HC�����@U@��A�{C�
                                    BxoH^  "          @����@�G�?��A{C(����@y��@A��C	�                                    BxoHl�  �          @�
=�xQ�@���>�?���C ���xQ�@��
?��AD  C�q                                    BxoH{Z  �          @��XQ�@�33�333��  B���XQ�@��
?z�@�{B��                                    BxoH�   
�          @���33@�p���
=�1G�Bۮ�33@�G�>��@��B��                                    BxoH��  
Z          @��1G�@��Y�����RB��1G�@�\)?�@��B��                                    BxoH�L  �          @��2�\@��R����33B�
=�2�\@�{?J=q@��B�G�                                    BxoH��  �          @�{�(Q�@�����p��^{B��(Q�@�
=?�G�A�B�Q�                                    BxoHĘ  
�          @�  �C�
@��
��ff���
B�L��C�
@�=q?fffAp�B���                                    BxoH�>  "          @�����@�(��8Q�����B�����@�(�?B�\@��
B���                                    BxoH��  
�          @�(��u@��׿#�
����B����u@���?�R@���B�Ǯ                                    BxoH��  
�          @Ǯ��
=@:=q��(���=qC޸��
=@O\)�p���\)C#�                                    BxoH�0  
�          @�
=��p�@
=����(�C� ��p�@%��˅�l��C��                                    BxoI�  �          @���ff@/\)����p�C@ ��ff@G
=��{�%C#�                                    BxoI|  T          @�  ���@C33�{���C&f���@`�׿���O�CxR                                    BxoI+"  
�          @�G����H@g
=������CǮ���H@�=q����A��C
p�                                    BxoI9�  
�          @�G���G�@K���(���z�CǮ��G�@dz῏\)�$z�C��                                    BxoIHn            @�z���{@;������Q�C�\��{@Y�����H�T(�C�                                    BxoIW  �          @�Q����@G��
=��ffC����@\)����d��C�f                                    BxoIe�  �          @�����
=?�(��޸R��=qC#@ ��
=@ff��G��:�HC��                                    BxoIt`  �          @�z����?�  ����%��C#�R���?�p��+���G�C!�R                                    BxoI�  �          @ʏ\��=q@fff����\)C����=q@p  ���Ϳs33C�                                    BxoI��  �          @�����33@~�R�"�\����C�
��33@�����H�X��Cff                                    BxoI�R  �          @�p����\@`���
�H��  C����\@|(���(��<z�C	��                                    BxoI��  �          @�z�����@W��\�hz�C�H����@i���!G���ffC                                    BxoI��  �          @�
=��\)@n�R��  �_33C� ��\)@~�R��\��Q�C��                                    BxoI�D  �          @�G�����@U� ����p�C� ����@n�R��{�"�RC��                                    BxoI��  �          @ƸR��@G
=�<�����C���@p  �z���\)C                                    BxoI�  T          @ȣ��\)@c�
�K����C
E�\)@�  ����(�CB�                                    BxoI�6  �          @ə��q�@qG��Tz����C��q�@���G����C�                                    BxoJ�  �          @Ǯ����@QG���\)��C�����@e��@  ���C��                                    BxoJ�  �          @�{���\@E<#�
=�\)Cٚ���\@?\)?J=q@�\C�                                    BxoJ$(  �          @����Q�@HQ���uC:���Q�@A�?B�\@��C                                      BxoJ2�  �          @�ff��p�?�\@j�HB=qC.���p���(�@k�B�C9�                                    BxoJAt  �          @�p��g
=>��@��BXG�C1�\�g
=��{@��BP�CE\                                    BxoJP  �          @�Q����@p�@\)A�G�Ch����?�@1�A�{C�R                                    BxoJ^�  �          @�����z�@.{��p���{C8R��z�@@�׿@  ���C��                                    BxoJmf  T          @����^�R@XQ��/\)�뙚C�
�^�R@}p���ff��z�CL�                                    BxoJ|  �          @�Q���ff@$z�?�  Ag�C����ff@
=@ffA�  C�                                    BxoJ��  �          @��R���@C�
?��@��\C�)���@3�
?�{AS33C                                      BxoJ�X  �          @�{��=q@P��?��A"�RC޸��=q@8��?�A���C�                                    BxoJ��  �          @�����R@i�������z�C:����R@dz�?G�@�Cٚ                                    BxoJ��  �          @�ff���\@xQ쿚�H�;�
C
)���\@�녾8Q��\C��                                    BxoJ�J  �          @��
���@QG�������Cff���@S�
>�\)@1G�C
                                    BxoJ��  �          @������R@U����
�\)C�����R@g
=�#�
�љ�CaH                                    BxoJ�  �          @��
�u�@9��?�R@�\C��u�@(��?�=qA�
Ch�                                    BxoJ�<  �          @����z�?Q�@�(�B�ǮC u��z�#�
@���B��fCCff                                    BxoJ��  �          @�Q��G�?�p�@�B�{Cn�G���  @���B�{C:aH                                    BxoK�  �          @�  �#33?��@�G�Bl{CY��#33>�@��HB�G�C)G�                                    BxoK.  �          @ə��0��?�Q�@�G�Be��C��0��?�@��B}��C)Q�                                    BxoK+�  �          @�(���?��@��B�CǮ��    @�Q�B�ǮC4�                                    BxoK:z  T          @�z����?���@\B���Cp����;��H@��B���CE5�                                    BxoKI   �          @˅�Q�?��@�\)Bt  C�)�Q�>�
=@���B���C*�                                    BxoKW�  �          @���E�@�R@���BI�\C5��E�?�@�Q�Bg��C=q                                    BxoKfl  T          @�33�B�\@E�@���B<ffC�H�B�\?��
@�G�Ba\)C�f                                    BxoKu  �          @ə��P��@$z�@��HB?z�C�P��?�ff@��HB]�RC5�                                    BxoK��  T          @�p���G�@5�@i��B
=C�R��G�?�ff@�Q�B/�C                                    BxoK�^  �          @�G��y��@
=@c33B�\C�
�y��?�{@���B1G�C ��                                    BxoK�  �          @������?�z�@<��A뙚C�����?�=q@UBp�C'k�                                    BxoK��  �          @��R�\(�@Z�H@J=qB=qC��\(�@\)@{�B+G�C�                                    BxoK�P  �          @�����?�Q�@QG�B33C�f���?��\@j=qB  C'!H                                    BxoK��  �          @�  ��{@��@W�B�RCG���{?�Q�@w�B$��C!�                                    BxoKۜ  �          @�Q���Q�?�p�@y��B$z�C� ��Q�?fff@���B8�RC'^�                                    BxoK�B  �          @��_\)@�
@?\)B��Cs3�_\)?�(�@Z�HB+�C �\                                    BxoK��  �          @�=q����aG�@aG�Bp�C6ٚ�����\)@UBQ�CBp�                                    BxoL�  T          @��H��\)@(Q�@�A�G�C޸��\)?�p�@2�\A�Q�CJ=                                    BxoL4  �          @��
����@:=q?��
A��C�f����@@{A�p�C�{                                    BxoL$�  T          @�  ��p�@^�R?���AL��CJ=��p�@AG�@�A��C!H                                    BxoL3�  �          @�\)���
@�33?�\)A.=qC+����
@j�H@	��A�=qC
\)                                    BxoLB&  T          @�(��_\)@�p�>�
=@�33B��\�_\)@���?�\)A���C p�                                    BxoLP�  T          @����q녿Tz�@��HB:Q�C@\)�q녿�\)@n�RB&  CNO\                                    BxoL_r  �          @�����(����@�33B2��C7����(�����@x��B'��CE�                                     BxoLn  �          @��R����#�
@�p�B2(�C6(������  @\)B(=qCDh�                                    BxoL|�  �          @�{��p��8Q�@y��B%z�C6W
��p�����@n{B�CC&f                                    BxoL�d  �          @�����R��(�@!G�A�=qC8����R��=q@z�A��\C?�q                                    BxoL�
  �          @����
=��\)@G�B��CD5���
=���@)��A�z�CLn                                    BxoL��  �          @�33��녿O\)@c33B�HC>{��녿�(�@Mp�B�\CH�H                                    BxoL�V  �          @��
��G��B�\@z=qB'�C=����G���G�@dz�B\)CJG�                                    BxoL��  �          @�z���=q�L��@k�BG�C6�=��=q��@`  B33CBT{                                    BxoLԢ  �          @�ff��  <��
@�(�B0C3���  ��=q@\)B)33CB33                                    BxoL�H  �          @�ff��{?   @eB
=C-�)��{��@eBQ�C9��                                    BxoL��  �          @��R��\)?L��@'
=A��
C+G���\)=��
@.{A�=qC3
                                    BxoM �  �          @����ff@=q?�(�A��
C+���ff?�{@33A���C J=                                    BxoM:  �          @������H@(�?�\)AR�HCs3���H?��R?�(�A��HC��                                    BxoM�  �          @�����\@{��z��1G�CL����\@{>��R@=p�CQ�                                    BxoM,�  �          @�����{@`��>�z�@2�\C5���{@S33?��HA>ffC�{                                    BxoM;,  �          @������@�  ?c�
A
{C	�����@g
=?�A�G�C��                                    BxoMI�  �          @Å����@�z�?�  A<��C
����@i��@�
A��
C�H                                    BxoMXx  �          @���tz�@���?��AA�CaH�tz�@�z�@\)A���C�3                                    BxoMg  �          @�(���
=@a�?�A|z�C5���
=@=p�@#33A���C�3                                    BxoMu�  �          @�z���  @g�?�AV{C�R��  @Fff@A��C�
                                    BxoM�j  �          @�33���@U?�p�A�\)C+����@0��@#�
A�ffC&f                                    BxoM�  �          @�Q����R@`��?�p�AZ�RC����R@?\)@�A�  C�                                    BxoM��  �          @��H��Q�@fff?�33AL(�CL���Q�@E@z�A��CT{                                    BxoM�\  �          @���z�@Z=q?��Ah  C)��z�@7�@=qA��
C�)                                    BxoM�  T          @Ǯ��
=@j=q?B�\@�  C�f��
=@Tz�?�(�A�=qCG�                                    BxoMͨ  �          @�
=����@fff?   @���Cu�����@U�?��HAYG�C��                                    BxoM�N  �          @�
=��ff@p��=�Q�?G�C�=��ff@fff?���A#
=C�                                    BxoM��  �          @ƸR��{@n�R�\)���C����{@hQ�?aG�A{C�R                                    BxoM��  �          @�\)����@y���(����(�C�R����@|(�>�
=@z=qC}q                                    BxoN@  �          @�{��\)@S�
=�Q�?Y��C���\)@J=q?}p�A�HC�)                                    BxoN�  �          @��
��Q�@^{>�\)@'�CG���Q�@QG�?��HA7�C�                                    BxoN%�  �          @�����z�@~�R>�@�Q�CY���z�@l��?�ffAj=qCk�                                    BxoN42  �          @�p�����@��R?#�
@�G�C������@��
?�A��C5�                                    BxoNB�  �          @ȣ��l(�@��?�(�A�ffC W
�l(�@|��@J�HA���C�                                    BxoNQ~  �          @ȣ���z�@��ͿO\)��z�C�H��z�@��R>�Q�@VffC0�                                    BxoN`$  �          @��H��\)@|�Ϳ�z��N�\C(���\)@���\)�   C
�                                    BxoNn�  �          @�\)����@w��J=q�陚C�3����@{�>��R@5�Cu�                                    BxoN}p  �          @�=q���@c�
�#33����C����@�(���p��X(�C	ٚ                                    BxoN�  �          @�p���{@l(������G�C:���{@�=q�L����33C	u�                                    BxoN��  �          @�����Q�@e��G��c\)C���Q�@w�����G�C�H                                    BxoN�b  �          @�G���@hQ�����(�C���@�Q�Q�����C	ٚ                                    BxoN�  �          @����Q�@u������G�CE��Q�@�녿�{�+
=C�                                    BxoNƮ  �          @����^{@���\)��ffC���^{@��H���\�G�
B�(�                                    BxoN�T  �          @�=q�j=q@[��9����=qC޸�j=q@�33��=q���C                                    BxoN��  �          @�
=�j�H@^�R�\)���C���j�H@~{��Q��F�\C                                    BxoN�  �          @�Q�����@O\)�����Ap�C�����@\�;�  ��RC�
                                    BxoOF  �          @�ff��33@Z�H�L�Ϳ�C���33@S33?fffA�
C��                                    BxoO�  T          @�
=��(�@n�R�8Q��Q�C#���(�@hQ�?^�RA
=C�f                                    BxoO�  �          @�ff��z�@Z�H>��@tz�Cff��z�@J�H?���AICc�                                    BxoO-8  �          @�G����
@Mp�?��@��RC
���
@:�H?�(�AX��CxR                                    BxoO;�  �          @�G���=q@8Q�?�\A���C�
��=q@G�@ ��A���C�                                    BxoOJ�  �          @�Q���G�@C33?��AL��C\��G�@!�@(�A��Cff                                    BxoOY*  �          @Ǯ��@1G�@
�HA�z�C����@33@7�A��Cc�                                    BxoOg�  �          @�Q����H@��@,��A�ffC=q���H?˅@Q�A���C"��                                    BxoOvv  �          @�  ��@#33@��A�{CǮ��?�\@AG�A�RC!)                                    BxoO�  �          @�Q���G�@�H@�
A�p�Cp���G�?�z�@9��A���C"�                                    BxoO��  �          @Ǯ��{@z�?��HA��\C����{?�@"�\A�{C"��                                    BxoO�h  �          @�  ���@-p�@�A�{C�����@G�@-p�A�z�C��                                    BxoO�  �          @����H@333@�A�=qC:����H@ff@/\)AӮC��                                    BxoO��  �          @\���@4z�@#�
A�G�C@ ���?�(�@P  B�\CaH                                    BxoO�Z  �          @�z�����@=q@@  A�ffC@ ����?���@dz�B
=C#{                                    BxoO�   T          @�Q����@(Q�@1�A�33CW
���?�(�@Z�HB
��C�                                     BxoO�  �          @�(���33@9��@*�HA�=qC^���33@ ��@XQ�B�
C.                                    BxoO�L  �          @�(�����@K�@*�HAظRC�
����@�@^{B�CB�                                    BxoP�  �          @�
=�y��@c33@2�\A�G�C	���y��@&ff@l(�B�CQ�                                    BxoP�  �          @�p��hQ�@u�@(Q�Aә�Ck��hQ�@:�H@g�BQ�C.                                    BxoP&>  �          @�
=�fff@vff@1G�A�{C!H�fff@8��@p��B�RCL�                                    BxoP4�  �          @�\)�S33@�z�@-p�A�z�C �\�S33@L(�@r�\B
=C                                    BxoPC�  �          @���^{@���@
=A�\)C �q�^{@Z�H@_\)B��Cc�                                    BxoPR0  �          @��H�[�@s�
@QG�B33C
=�[�@,��@�\)B0ffC��                                    BxoP`�  �          @Å�U@g�@fffB�C���U@�@��B=��C                                    BxoPo|  �          @��H�J=q@�{@E�A���B���J=q@G�@��B,z�Ch�                                    BxoP~"  �          @����Tz�@�
=@,��A�C 33�Tz�@P  @s�
BffC�
                                    BxoP��  �          @��Z=q@u�@\(�B�C�f�Z=q@*=q@��B6(�C��                                    BxoP�n  
�          @�{�I��@]p�@�G�B#33C\)�I��@Q�@��
BP  C�                                    BxoP�  �          @�p��L��@u�@^�RB��C��L��@(��@�ffB;��CxR                                    BxoP��  �          @�33�=p�@2�\@�G�B@p�C���=p�?��@���Bg{C��                                    BxoP�`  �          @\�*�H@^{@�ffB/\)B�\�*�H@@�G�B`Q�C�                                    BxoP�  �          @ƸR�%�@g
=@��\B1=qB�
=�%�@�@�ffBd  C                                    BxoP�  �          @�{�!�@l(�@�  B.
=B����!�@�@�z�Ba��C
�                                    BxoP�R  �          @ƸR��p�@`  @�=qBK�
B�p���p�?�
=@�z�B���CǮ                                    BxoQ�  �          @Ǯ��ff@S�
@��RBS33B�=q��ff?ٙ�@�
=B�(�C                                    BxoQ�  �          @�  ��@b�\@�BC�\B�q��@   @���Bz��Cu�                                    BxoQD  �          @�
=�*=q@n{@�
=B*�B�G��*=q@33@�(�B^
=C(�                                    BxoQ-�  �          @ƸR�8Q�@U�@��B2�RC�
�8Q�?��@���B`��C�3                                    BxoQ<�  �          @ƸR�$z�@~�R@�Q�B �HB����$z�@'
=@�Q�BW=qC�                                     BxoQK6  �          @��
��G�@\��@�33BU��Bԙ���G�?���@���B��B�#�                                    BxoQY�  �          @Å��Q�@XQ�@�G�BEffB�q��Q�?�\)@��HB|C                                      BxoQh�  �          @�(��ff@i��@�33B7p�B��H�ff@(�@��Bo�Cٚ                                    BxoQw(  �          @�33���R@���@���B2Q�B�=q���R@,(�@�=qBs��B�                                    BxoQ��  �          @�{��G�@�(�@�B*  B����G�@;�@���Bk��B�\                                    BxoQ�t  �          @�Q���H?��H@��HB���B��=���H�.{@�p�B��\CiǮ                                    BxoQ�  �          @Ǯ�5?���@�(�B��B�{�5=�Q�@�{B�ǮC,xR                                    BxoQ��  �          @�(��8Q�?�\)@�=qB�(�B�
=�8Q�=#�
@ÅB��=C)
=                                    BxoQ�f  �          @�z�k�?�\)@�Q�B��=B�33�k��@  @�=qB�Q�C|��                                    BxoQ�  �          @���?8Q�?���@�
=B��Bc�?8Q�@  @���B�ffC��H                                    BxoQݲ  �          @�����\?�\)@���BfffC�{��\>�p�@�  B�8RC*��                                    BxoQ�X  �          @�(�����@0��?���A6{C�����@��?�p�A�C�H                                    BxoQ��  �          @�ff��{@%?�AT(�C����{@�\@��A�Cs3                                    BxoR	�  �          @�(����@!�?�A|��C�����?�z�@
=A�33C E                                    BxoRJ  �          @�����
=@�H?�\)A��
C#���
=?�p�@ ��Aď\C!�)                                    BxoR&�  �          @�=q��{@Q�@�\A��C\)��{?�33@*�HAиRC"h�                                    BxoR5�  �          @Å��G�@(�?�A�C8R��G�?�\@{A���C!xR                                    BxoRD<  �          @�33��{@(��?��A�z�C��{?�(�@ ��A£�C=q                                    BxoRR�  �          @��
��=q@%�@  A��RC
=��=q?�\@<(�A�G�C �\                                    BxoRa�  �          @�����@��@p�A���C���?��@AG�A�p�C%s3                                    BxoRp.  �          @����(�@��@�\A��
C0���(�?���@8Q�A��HC$:�                                    BxoR~�  �          @��R��z�@�?���A���C���z�?���@p�AĸRC"��                                    BxoR�z  �          @�����33@z�?��HA�z�C�{��33?�Q�@��A��C$�f                                    BxoR�   �          @�{����@�?���AX(�C#�����?��@�\A��HC!B�                                    BxoR��  �          @�  ����?^�R?�=qA'�C+u�����>��H?�ffAI��C/&f                                    BxoR�l  �          @�
=��
=?���?�A�(�C%(���
=?O\)@��A���C+��                                    BxoR�  �          @������@�?�\)AzffCh����?�
=@
�HA��C%5�                                    BxoRָ  �          @�
=��z�@<��?�  A�=qC�)��z�@��@%�A���C.                                    BxoR�^  
�          @�����=q?�?�(�A��C (���=q?��H@{A���C'33                                    BxoR�  �          @�=q���\@!G�@Q�A��RC�����\?�p�@3�
A�G�C!33                                    BxoS�  �          @��
����@@�
A�33C
����?���@,(�A�Q�C#^�                                    BxoSP  �          @�����=q@�\?�{A���C
=��=q?���@=qA�G�C%�q                                    BxoS�  �          @�\)���?Y��?�A5C+�=���>�G�?���AW
=C/�{                                    BxoS.�  �          @��R��p�=L��?(��@˅C3���p��#�
?#�
@�{C5�\                                    BxoS=B  �          @�����#�
?z�@��\C4^����aG�?
=q@�C6)                                    BxoSK�  �          @�����?��R?O\)@�p�C'������?k�?�z�A6�\C*��                                    BxoSZ�  �          @��
���>����R�@��C/z����?������C.                                    BxoSi4  �          @���
=@
=q>�@�{Cp���
=?�33?�\)A.�HC ��                                    BxoSw�  �          @�����p�?����\�j�HC%���p�?��R=���?�G�C%B�                                    BxoS��  �          @�{��Q�?��R?J=q@�
=C (���Q�?У�?��AX��C#��                                    BxoS�&  �          @�����H@,��?^�RA��C5����H@�?�Q�A��C�f                                    BxoS��  �          @����@&ff?�\@�\)C����@33?��AD��C�                                    BxoS�r  �          @�����@�
>���@/\)C k�����?�{?p��AQ�C"8R                                    BxoS�  �          @ƸR��ff?�p�>��@p  C#���ff?�G�?p��A\)C%                                    BxoSϾ  �          @�  ��(�?��R?��@��C!G���(�?��H?�z�A+�
C#Ǯ                                    BxoS�d  �          @�p���(�?�(�?��@��C#����(�?���?���A z�C&#�                                    BxoS�
  �          @�33��ff?��?�R@�(�C)�)��ff?Tz�?p��A�\C,
=                                    BxoS��  �          @��H��p�?xQ�?Y��A ��C*�f��p�?&ff?�\)A)�C-�                                     BxoT
V  �          @�����?У�?�G�A�33C#�3���?xQ�@(�A�G�C*E                                    BxoT�  �          @�����R?��H@{A��
C"�����R?fff@*=qA�G�C*�)                                    BxoT'�  �          @��
����?��
?�
=A�p�C"#�����?�ff@��A�\)C).                                    BxoT6H  �          @�����?�?�{A�
=C#h����?z�H@33A�(�C*#�                                    BxoTD�  �          @�����=q?���?��HA�G�C �q��=q?��
@  A�  C'�                                    BxoTS�  �          @���{@\)?���A���C�{��{?��
@��A��
C$Q�                                    BxoTb:  �          @�������@-p�?�(�A^ffC������@�@�A�z�C��                                    BxoTp�  �          @�����33@\)?�p�A<��C���33?���?�p�A��C�R                                    BxoT�  �          @�33��ff@�?�=qAs33C{��ff?���@�A�33C%!H                                    BxoT�,  �          @�{���@�?�G�A�=qC����?��@�A��C&p�                                    BxoT��  �          @ƸR���\?�@ ��A�\)C"{���\?��@\)A��C)k�                                    BxoT�x  �          @�
=��{@�\?�ffA��C E��{?�=q?�z�A{33C$xR                                    BxoT�  �          @�=q����@(�?���A/�CE����?�?�{A��C#�f                                    BxoT��  �          @�=q���
@ff?z�HA�RC \)���
?�33?�\)Am��C$E                                    BxoT�j  �          @�=q��G�@ ��>#�
?�Q�C����G�@z�?z�HA�C.                                    BxoT�  �          @�G�����@-p�<#�
=uC\)����@#�
?h��AG�C��                                    BxoT��  T          @�G����@<�Ϳ����K�C(����@P  ���
�:�HC��                                    BxoU\  �          @������@.�R����  Cٚ���@:�H��\)�(�CL�                                    BxoU  �          @�(�����@S33�����{CT{����@J=q?xQ�Ap�Cs3                                    BxoU �  �          @�(����R@L(��W
=��(�C}q���R@E�?^�RA
=Ck�                                    BxoU/N  �          @��H��{@+�<#�
>\)C�q��{@!G�?h��A
�RC�                                    BxoU=�  �          @Å��@2�\=�Q�?^�RC�\��@&ff?��
A�RCs3                                    BxoUL�  �          @�  ��Q�@�\����  CJ=��Q�@�>�z�@(��C�3                                    BxoU[@  �          @�  ����@{�
=��p�C�����@�\>B�\?��
Cz�                                    BxoUi�  �          @�  ����@�R��
=�z=qC�����@\)>�33@QG�C�{                                    BxoUx�  �          @�Q����?��?��A z�C"0����?�33?�z�Av�RC&��                                    BxoU�2  �          @�Q�����?�?��RA]�C&s3����?Tz�?��A��\C+��                                    BxoU��  T          @��H����?�ff?�Q�AR�\C'Ǯ����?=p�?�ffA�{C-                                    BxoU�~  T          @�G����R?�Q�?�=qA=qC$0����R?�(�?˅AjffC(p�                                    BxoU�$  �          @�G���\)?��?�=qAD��C'�=��\)?B�\?�Q�A{33C,�q                                    BxoU��  �          @ƸR���@논��
�.{C���@��?B�\@��C�                                    BxoU�p  �          @�  �P��@L(�����+�HC���P��@�ff�9�����
B�u�                                    BxoU�  �          @ʏ\�=p�@K����H�:Q�C�3�=p�@���N{��G�B���                                    BxoU��  �          @����@��@J�H��  �7(�C���@��@����HQ���B�W
                                    BxoU�b  �          @ə��5�@j�H����'�\B���5�@�(��(���ȣ�B��                                    BxoV  �          @ȣ��!�@�Q��l�����B�Q��!�@����   ��33B�.                                    BxoV�  T          @�=q�7
=@�G��j=q�ffB�� �7
=@�G��������
B��f                                    BxoV(T  �          @ʏ\�Dz�@\)�tz���
B�B��Dz�@�=q�p����B�                                    BxoV6�  �          @��H�W
=@p���w
=�C�\�W
=@���z���G�B�L�                                    BxoVE�  �          @ə��I��@��\�e�G�B�=q�I��@�=q��Q���=qB�                                    BxoVTF  �          @˅�X��@k��z�H�\)C���X��@�������33B�\)                                    BxoVb�  T          @��H�X��@q��p  �C���X��@�33�������B��                                    BxoVq�  �          @ʏ\�b�\@aG��x�����C.�b�\@����H��B��                                     BxoV�8  �          @˅�X��@Y����=q�!G�C��X��@���(Q���\)B���                                    BxoV��  �          @�=q�y��@r�\�I����RC���y��@����\)�o�C��                                    BxoV��  �          @�33��{@I���O\)��{C�H��{@��H��
=���C	\)                                    BxoV�*  �          @����(�@`���C33���CJ=��(�@����\)�j�RC&f                                    BxoV��  �          @���Q�@K��:=q��\)CE��Q�@�  �����g
=C�R                                    BxoV�v  �          @�33���H@L���E��Q�CxR���H@��H��G��x��CǮ                                    BxoV�  �          @љ���p�@!G�������CT{��p�@s�
�<���ָRCB�                                    BxoV��  �          @������@%��}p���CQ�����@tz��2�\��\)C�f                                    BxoV�h  �          @Ӆ��{@{��ff�(�C����{@s33�C33��(�Cp�                                    BxoW  �          @�\)��{@���33��RC���{@e�AG���33C�                                    BxoW�  
�          @�{�n�R@(�����\�-C���n�R@�Q��G
=��Q�C��                                    BxoW!Z  
(          @�  �<(�@������
�Q�B�W
�<(�@�\)�=q��\)B�R                                    BxoW0   �          @�Q��U�@=p���
=�:p�C
aH�U�@�ff�Vff��ffB���                                    BxoW>�  �          @θR�\)@n{���H�6�
B����\)@��
�<����p�B��                                    BxoWML  �          @Ϯ�ff@J�H��(��Q�
B�33�ff@����j=q�	�B�=q                                    BxoW[�  �          @љ����@h����ff�Dz�B����@�p��S�
���B�=q                                    BxoWj�  �          @�  �
�H@|(����8��B���
�H@���<����(�B�\                                    BxoWy>  �          @�\)��p�@w
=�����>��B�.��p�@�=q�C�
��\BܸR                                    BxoW��  �          @�{��ff@�z�����:{B���ff@�G��6ff����B�(�                                    BxoW��  �          @�{�\(�@�(�����7��B�Q�\(�@�  �-p����
B�{                                    BxoW�0  �          @θR�c�
@�\)��Q��2��Bʊ=�c�
@���%�����B�u�                                    BxoW��  �          @�  �AG�@J�H��G��833C���AG�@�33�E����
B��{                                    BxoW�|  �          @�����33?�z��~�R��C#����33@1G��L(���{CL�                                    BxoW�"  T          @�\)�����z�H�Z�H��C>}q����>����b�\�C0k�                                    BxoW��  T          @Ϯ��
=��\)�K���
=CB���
=���]p��{C55�                                    BxoW�n  �          @θR��=q���^{��C@p���=q>8Q��i���	��C2)                                    BxoW�  �          @�
=��
=�����j�H�
(�C?����
=>�33�s�
��C0+�                                    BxoX�  �          @Ϯ��
=����j�H�	��C@Y���
=>�\)�u���HC0��                                    BxoX`  �          @����Ϳ��R��Q���
CBB�����>��
��{�!��C0G�                                    BxoX)  �          @����{��ff��=q�(�CFB���{=u����+��C38R                                    BxoX7�  �          @�p�������������� �CI�\�����#�
�����3Q�C6�                                    BxoXFR  �          @���=q���\�����\CB�q��=q>��
����%
=C00�                                    BxoXT�  �          @����H��p������p�CB8R���H>�33��ff�#G�C/�)                                    BxoXc�  �          @�����Ϳ�{�y���  CC�=����>8Q���(�� ffC1޸                                    BxoXrD  �          @�
=��z�c�
�s�
��C=�)��z�?
=�w���C-�                                     BxoX��  �          @�ff�����
=����ffCA�����>�
=���R�#=qC/�                                    BxoX��  �          @�p���녿�(��}p����CD�����=���
=�%�C2�)                                    BxoX�6  
�          @ʏ\�������H�z=q�33CKJ=������Q�����.C8xR                                    BxoX��  �          @�G���Q��33������
CPT{��Q�!G����\�;=qC<Y�                                    BxoX��  �          @����p��У����R�,��CIY���p�=u��Q��<�C3)                                    BxoX�(  �          @��H��(��(�����:=qC<c���(�?������5(�C%33                                    BxoX��  �          @�=q�u�aG������I�RC7T{�u?����z��:�C�                                    BxoX�t  �          @�����\�   ��=q�3(�C:�����\?�z���ff�,z�C$��                                    BxoX�  �          @��H���R����33�!��CP����R�(����>�
C<0�                                    BxoY�  �          @�������0���k��
=CU
�������ff�5\)CCc�                                    BxoYf  �          @�����G���\����+�HCK�q��G���Q���
=�?C5=q                                    BxoY"  �          @��R���H���R�tz����CM�3���H�Ǯ��G��8��C9��                                    BxoY0�  �          @�G���=q�z��x���!CO���=q��G���z��<  C:(�                                    BxoY?X  �          @��L�;.{����jG�C6�q�L��?�����TG�C{                                    BxoYM�  T          @�\)�L�;u����kp�C8=q�L��?���  �VC�\                                    BxoY\�  �          @���g
=��33��ff�V�C9���g
=?������R�HQ�C�\                                    BxoYkJ  �          @�zῼ(�?}p����\L�C\��(�@H����Q��]=qB�                                      BxoYy�  �          @�p���z�?Q���{33Cٚ��z�@B�\��p��dG�B���                                    BxoY��  �          @�����?s33��33C:����@HQ���G��\�\B�#�                                    BxoY�<  �          @��Ϳ޸R?#�
�����C�H�޸R@5��p��d�RB��H                                    BxoY��  �          @�p���?L�����\�{C5ÿ�@>�R���\�^
=B�p�                                    BxoY��  �          @��
��Q�?   ������C%ff��Q�@,(���(��d�B�ff                                    BxoY�.  �          @�z��z�>�  ���W
C,uÿ�z�@�R��Q��m
=B�.                                    BxoY��  �          @��H����<���G��C3  ����@G����\�uffC �                                    BxoY�z  �          @��H��\)?@  ��G�C33��\)@<(������a��B�                                    BxoY�   �          @�Q���>\��\)B�C&޸���@%���(��l�B��H                                    BxoY��  �          @�=q��(�=#�
���C2�R��(�@33���H�v��B���                                    BxoZl  T          @�Q���>8Q�����C.n���@������m��C s3                                    BxoZ  T          @�{�z�B�\��{�fCH��z�?�
=����=qCc�                                    BxoZ)�  �          @�ff�1�?����\)�u��C'���1�@&ff���\�I
=Cٚ                                    BxoZ8^  �          @����?c�
���
33Cp����@;����H�Lp�B�z�                                    BxoZG  �          @�{�7
=?(�����q��C(\�7
=@%���  �E�RC	�q                                    BxoZU�  �          @�ff�3�
?z���ff�t\)C(h��3�
@$z������H=qC	��                                    BxoZdP  �          @�Q��c33?����\)�O�C"�3�c33@7
=�z�H�"�HC{                                    BxoZr�  �          @�\)�k�?��������FG�C!��k�@9���mp��Q�C�\                                    BxoZ��  �          @�\)�aG�?�Q����N33C!E�aG�@<���u��\)C�                                    BxoZ�B  �          @�\)�q�?�
=�����4\)C�q�q�@L(��H����
Cٚ                                    BxoZ��  �          @���c33@2�\�n{��RC�{�c33@�G��33����CY�                                    BxoZ��  �          @�ff�p  @�\��(��0{Cz��p  @`���>�R����C��                                    BxoZ�4  
�          @�{�fff@(���(��1��C���fff@j=q�:�H��33C��                                    BxoZ��  T          @����O\)@8����  �*z�C
B��O\)@��� ���ɮB��R                                    BxoZـ  �          @�p��%�@:=q��(��A�C��%�@�p��7
=��B��                                    BxoZ�&  �          @�33��@!���ff�W�C�
��@�ff�R�\�	��B�                                    BxoZ��  �          @��\����@{�����p�C����@���o\)��B���                                    Bxo[r  �          @���ٙ�?�ff��G��
CaH�ٙ�@qG���(��3�B�\                                    Bxo[  �          @��
��@
=q�����u=qC�)��@����vff�#��B�{                                    Bxo["�  �          @��
���
@z���
=�|\)B����
@�
=�u�$G�Bճ3                                    Bxo[1d  �          @�(��G�@����`�C{�G�@�33�`  �=qB��)                                    Bxo[@
  �          @�33�.�R@0  �����?��CǮ�.�R@�  �5���B�ff                                    Bxo[N�  �          @��\�333?��
��(��j�C�\�333@:=q��G��4{C޸                                    Bxo[]V  �          @�z��4z�?   ����s  C*  �4z�@!���
=�G
=C
(�                                    Bxo[k�  �          @�(��=p�?��
���f\)C ޸�=p�@<(����H�1�HC33                                    Bxo[z�  T          @�z��
�H@=q��(��a�C��
�H@��^�R�B��                                    Bxo[�H  �          @����5@a��k��33C ���5@�
=�����
B�
=                                    Bxo[��  �          @����Fff@n{�E����C�q�Fff@����ff�QG�B�=q                                    Bxo[��  �          @�
=�(��?�Q���{�`�
C^��(��@]p��e�{B�                                    Bxo[�:  �          @�(��p�@(���  �Wz�C��p�@�=q�G
=�B��                                    Bxo[��  �          @��Ϳ��@C�
���\�I{B�33���@�=q�,(�����B��)                                    Bxo[҆  �          @�p��@33��
=�bffC
��@s33�]p���B�.                                    Bxo[�,  �          @����(Q�?�ff����jp�C�q�(Q�@J=q�vff�,�C��                                    Bxo[��  �          @�z��G�@@���_\)�=qC{�G�@���Q���\)B��{                                    Bxo[�x  �          @�(��$z�@�����R�TQ�C33�$z�@r�\�K��
=B�\)                                    Bxo\  �          @�����?�������j(�C���@g
=�h��� ��B��R                                    Bxo\�  T          @�{�:=q@�H��Q��A��C@ �:=q@}p��8Q�����B��                                    Bxo\*j  
�          @�
=�?\)@$z���z��9C@ �?\)@���-p���{B��                                    Bxo\9  
�          @�  �:=q@33���
�F�C���:=q@z=q�AG�����B�\)                                    Bxo\G�  �          @�G��1G�@=p���z��6p�C(��1G�@���"�\���
B�Q�                                    Bxo\V\  �          @���@  @����=q�C�HC�q�@  @w
=�?\)����B���                                    Bxo\e  �          @����(Q�@!����R�K
=C��(Q�@���@����p�B���                                    Bxo\s�  �          @��R�*�H?����p��\
=C���*�H@i���]p���B�aH                                    Bxo\�N  �          @�  �*�H?�G����R�p33C.�*�H@?\)���\�5��C��                                    Bxo\��  �          @����"�\?�z����iz�C��"�\@dz��q��#=qB���                                    Bxo\��  �          @����@1G������L��C�f��@���<����(�B잸                                    Bxo\�@  �          @�  �&ff@c33�i�����B���&ff@�  �������RB�k�                                    Bxo\��  �          @����C�
@<(��tz��'�HC8R�C�
@�����R���B�W
                                    Bxo\ˌ  �          @����N�R@:=q�r�\�$=qC
�N�R@�\)�p���ffB��R                                    Bxo\�2  �          @�\)�<��@���%���(�B�W
�<��@�녿(���(�B�                                      Bxo\��  T          @��R�(��@�(��G���\)B�(��(��@��W
=��B�W
                                    Bxo\�~  �          @����A�@'���(��7�C8R�A�@��
�(����=qB�                                    Bxo]$  �          @�=q�I��@S�
�c�
��C�=�I��@�Q�������RB��)                                    Bxo]�  �          @�G��6ff@G
=�vff�)z�C}q�6ff@�ff�
�H��Q�B�G�                                    Bxo]#p  �          @����*�H@8Q���\)�<�C޸�*�H@����'
=����B�                                     Bxo]2  �          @�ff�'
=@<���}p��4�RC�{�'
=@�33����p�B���                                    Bxo]@�  �          @��\��Q�@]p������9
=B�\��Q�@��H�
=q���B�u�                                    Bxo]Ob  �          @��Ϳ�\@S�
��{�@��B�(���\@�G��Q����B܊=                                    Bxo]^  �          @�p���(�@H����G��U�\Bޔ{��(�@����0�����
Bнq                                    Bxo]l�  �          @��
�h��@H������Y33B�(��h��@�G��1G����HB�p�                                    Bxo]{T  �          @�33���@2�\���R�f
=B����@�G��C�
��
B�{                                    Bxo]��  �          @�녿��@J�H���\�P\)B۞����@�\)�#33��p�B�33                                    Bxo]��  �          @�(��Q�@n�R�g
=� (�B�ff�Q�@���z���G�B�Ǯ                                    Bxo]�F  �          @����ٙ�@i���r�\�,��B��Ϳٙ�@�������  B���                                    Bxo]��  T          @�33��ff@[������H�\B�  ��ff@�ff�������B��                                    Bxo]Ē  �          @������@U��G��H  B�  ����@�����͙�B�                                      Bxo]�8  �          @�33���@J=q��
=�T�HB�(����@����*�H��=qBͽq                                    Bxo]��  �          @�Q��(�@Mp��q��0��B��{�(�@����G����RB癚                                    Bxo]��  �          @�
=���
@#�
���\�Y{B��Ϳ��
@��2�\���RB�.                                    Bxo]�*  �          @�ff?�\@=p���{�_�\B�p�?�\@��H�-p����B�Q�                                    Bxo^�  �          @��@{@`  �X���G�B[Q�@{@�(����
���
Bw�                                    Bxo^v  �          @�\)?fff@#33��\)�q�B�8R?fff@�33�I���G�B��R                                    Bxo^+  �          @�p�>���?�ff��G�B��>���@tz��p  �1=qB��                                    Bxo^9�  �          @��R���R@*�H��\)�^�RB�aH���R@���7���
=B�Ǯ                                    Bxo^Hh  
�          @�ff��{@	����(��iC���{@}p��N{�z�B�=q                                    Bxo^W  �          @�\)�z�?u���\Q�B���z�@L(����
�U��B�u�                                    Bxo^e�  
�          @�
=>W
=?�G������B��R>W
=@u�vff�4(�B��H                                    Bxo^tZ  �          @�
=����@N{��
=����C.����@hQ���Ϳz�HC�{                                    Bxo^�   �          @����@N�R��33�7�Cn��@Z=q>\@q�C�R                                    Bxo^��  �          @�����z�@2�\���R���
C�R��z�@XQ�\)��33C�H                                    Bxo^�L  �          @�ff��?��H�$z���
=C8R��@9����Q��k
=C:�                                    Bxo^��  �          @�����33>Ǯ�:�H�(�C/#���33?���������C c�                                    Bxo^��  �          @�����z�?���#33��C&����z�@
=q���
���
C33                                    Bxo^�>  T          @��R��G�?�
=�  ���
C$5���G�@녿���`��C�R                                    Bxo^��  T          @��H��?�������C&�q��@�
������
=CQ�                                    Bxo^�  �          @����?�z��(���C&����@������C�R                                    Bxo^�0  �          @�����R?�Q���
����C'����R?��R�����T��C&f                                    Bxo_�  T          @�
=���?�p��У���G�C!xR���@녿0���޸RC8R                                    Bxo_|  �          @�G����
@
=q��G��s�C����
@'
=�Ǯ�x��C                                      Bxo_$"  T          @����z�?��ͿJ=q��
=C#u���z�?��
<#�
=���C!��                                    Bxo_2�  �          @��H���H?�ff�����Mp�C$xR���H?��>���@~�RC$��                                    Bxo_An  �          @��H���?�p������g�C"!H���@�Ϳ���ffC�R                                    Bxo_P  �          @��\��(�?�=q��p��C\)C!+���(�@(���\)�/\)C�
                                    Bxo_^�  �          @�p����\?�p�����LQ�C%#����\?�z�   ��z�C!�                                    Bxo_m`  �          @������?�z���
�o�C%� ����?�Q�:�H���
C ��                                    Bxo_|  �          @�33��p�?��ÿ����e�C#�{��p�@33�z���  C=q                                    Bxo_��  �          @�������@��
=��Q�Cu�����@=p��+���z�C�H                                    Bxo_�R  �          @�33���\@�� ����z�C���\@0�׿Y���{C��                                    Bxo_��  �          @����(�@�\��
=���C���(�@333�����=qCW
                                    Bxo_��  �          @��
����@=q��  ����C������@<(�������Cٚ                                    Bxo_�D  �          @�z���{@�R�   ���HCQ���{@G��(�����C�q                                    Bxo_��  �          @�(���=q@�R��\��p�CǮ��=q@P  �k���
C�                                    Bxo_�  T          @�33��G�@�\����G�C����G�@8Q������C@                                     Bxo_�6  �          @����=q@�H�G����HC\)��=q@L(��n{���C�                                    Bxo_��  �          @������@%�����C�)����@QG��333���C��                                    Bxo`�  �          @��R��G�@$z��=q���C��G�@X�ÿxQ��ffC��                                    Bxo`(  �          @�
=��Q�@/\)�z���G�C
=��Q�@`  �O\)���HC��                                    Bxo`+�  �          @�
=���@:�H�����\C\)���@aG�������HCW
                                    Bxo`:t  �          @�  ��@(Q��E�����CG���@p  ���
�lz�C                                    Bxo`I  T          @�\)���@8Q��+���G�CQ����@r�\����#�
C�)                                    Bxo`W�  �          @�Q���{@:=q�����=qC.��{@l(��L����z�C��                                    Bxo`ff  �          @�\)��(�@(��������C����(�@Vff�8Q��޸RC��                                    Bxo`u  �          @�
=���@G
=���R�g�
CW
���@\��=�G�?��
C�)                                    Bxo`��  �          @��R��@Q녿���p��C�q��@g�>��?�z�CL�                                    Bxo`�X  T          @��R����@Tzῗ
=�6=qC^�����@_\)>��H@�C�                                    Bxo`��  T          @�p���=q@.�R�Q���Q�C����=q@Y���!G��ÅC�=                                    Bxo`��  �          @�{��(�@R�\>��
@Dz�C  ��(�@1G�?�ffA�Q�Cc�                                    Bxo`�J  �          @�ff��ff@�z�?+�@�Q�Cc���ff@W
=@ ��A�ffCT{                                    Bxo`��  �          @�=q�\��@�?�
=A�z�C�=�\��@3�
@h��B(�C�\                                    Bxo`ۖ  �          @�ff�&ff@�p�?�A��B�.�&ff@S33@qG�B'�C :�                                    Bxo`�<  T          @�p��]p�@�33�B�\��C �\�]p�@|(�?���A��CY�                                    Bxo`��  �          @�������@\)�u�
=C������@)��>��R@G�C0�                                    Bxoa�  �          @�=q��G�@P  �����RC�=��G�@HQ�?�ffA&{C�)                                    Bxoa.  �          @��H����@S33�
=��=qCY�����@L��?��\A (�C33                                    Bxoa$�  �          @�=q���@Fff=�Q�?h��C�3���@,��?��
AtQ�C+�                                    Bxoa3z  �          @�(���@J�H>�33@Z=qC=q��@(��?��A�{C��                                    BxoaB   �          @������@#33?�=qA|Q�C�R���?�G�@%A�G�C#(�                                    BxoaP�  �          @�{�\��@@��@3�
A�=qC
���\��?��@x��B9�C��                                    Bxoa_l  �          @�(��z�@�{��z��P��B�#��z�@�
=?�ffA=p�B��
                                    Bxoan  �          @�=q�\)@�G���z���\)B�\�\)@��R>W
=@(�B�W
                                    Bxoa|�  �          @�=q��(�@�\������C����(�@?xQ�A%�C�)                                    Bxoa�^  �          @����@�>�=q@.�RC&f��?��H?��HAHQ�C!                                    Bxoa�  �          @�z�����@���p��q�C������@��?J=qAC��                                    Bxoa��  �          @��H����>�ff=u?�RC/T{����>�p�>�=q@/\)C0+�                                    Bxoa�P  �          @��R���R?�p�=�?��RC33���R?�Q�?��A,  C"�                                    Bxoa��  �          @�=q���@>{>u@z�CB����@   ?�{A���CY�                                    BxoaԜ  �          @��
��
=@*=q>�\)@.{C����
=@{?�G�Am�C                                      Bxoa�B  �          @��\����@�H>��H@��HCaH����?�33?�=qA{�C :�                                    Bxoa��  T          @��
���\@
=?z�@��C!H���\?�ff?У�A�
=C!O\                                    Bxob �  �          @�\)��(�@�?fffA
�HC����(�?��H?���A�{C"c�                                    Bxob4  �          @�  ��z�@"�\?333@�{CǮ��z�?�33?���A�\)C ��                                    Bxob�  �          @�
=��=q@'
=?G�@�Q�C����=q?�?�
=A��\C !H                                    Bxob,�  �          @��R���H@%�?#�
@�{C:����H?��H?��A�  C�
                                    Bxob;&  �          @�  ���@&ff?+�@�{C����?�(�?�=qA�Q�Cٚ                                    BxobI�  �          @��R���
@4z�?���A.�HC����
?�Q�@A��C8R                                    BxobXr  �          @�=q���@6ff?
=q@���C�����@\)?�A�p�C�                                    Bxobg  �          @�����33@+�>��@��Cn��33@Q�?�Q�A�
=CW
                                    Bxobu�  �          @����
=@#33�xQ��(�C����
=@,��>�p�@dz�C��                                    Bxob�d  �          @�Q���33@C�
>���@9��C  ��33@"�\?�p�A�\)Cz�                                    Bxob�
  �          @�Q���=q@H��>���@s33C@ ��=q@#�
?���A��\C0�                                    Bxob��  �          @�  ���\@E?
=@�z�C�����\@�@   A�z�Cn                                    Bxob�V  
�          @�G���=q@J�H?J=q@�  C���=q@��@p�A�=qC��                                    Bxob��  �          @�����@U�?\(�Ap�C�)���@   @
=A��RCǮ                                    Bxob͢  �          @�
=���
@j=q?O\)@��
C�����
@4z�@�RA�
=C�)                                    Bxob�H  �          @�
=��33@C�
>�33@Tz�C�q��33@!G�?��
A��C�R                                    Bxob��  �          @�ff�i��@�z�@(�A�G�Cff�i��@$z�@z�HB%(�C��                                    Bxob��  �          @��tz�@�p�?��HA���C���tz�@4z�@`��B��C�{                                    Bxoc:  �          @�����Q�@1�?���AS33C����Q�?��@!G�A���C ff                                    Bxoc�  �          @����=q@P  ?�  A�
C����=q@
=@��A�=qC�)                                    Bxoc%�  �          @�z�����@G�?aG�A	G�C������@33@�A���C�
                                    Bxoc4,  �          @����G�@@  ?
=@�
=C8R��G�@?�(�A�=qC�                                    BxocB�  �          @�p����R@Z�H?xQ�A33C�����R@!G�@ ��A�\)C�{                                    BxocQx  �          @�
=���H@-p�@%�A�z�Cp����H?�{@dz�B\)C&Y�                                    Bxoc`  �          @�����  @=q@.{A�  C#���  ?G�@b�\B\)C*��                                    Bxocn�  �          @�
=���R@.�R?���A���C(����R?�p�@<��A�33C#Y�                                    Bxoc}j  �          @�ff����@+�?��Ap��C�����?˅@)��A���C"��                                    Bxoc�  �          @����\@8Q�?��A�ffC(����\?˅@C33A��C!��                                    Bxoc��  �          @��R��Q�@)��?�A�C#���Q�?�@7�A��C$!H                                    Bxoc�\  �          @�ff��=q@{?��HAc
=Ch���=q?�(�@
=A���C')                                    Bxoc�  �          @���z�@(��@�
A��\C���z�?��@EA�
=C%8R                                    Bxocƨ  �          @���z�@.{@!G�A�z�C�
��z�?��@a�BC&33                                    Bxoc�N  �          @���G�@,��@��A�(�C�=��G�?��
@O\)B�RC%�                                    Bxoc��  �          @�p���z�@,(�?��
A�C.��z�?�(�@8Q�A�
=C#L�                                    Bxoc�  �          @�ff����@��@{A�\)C������?�  @G�A�p�C(��                                    Bxod@  �          @�\)��  @�R@�
A�(�C�)��  ?��@@��A�C')                                    Bxod�  �          @�Q���z�@ ��=�\)?!G�C���z�@	��?��AI�C:�                                    Bxod�  �          @�Q����\@,��>��@���C.���\@
=?޸RA���Cc�                                    Bxod-2  
�          @�  ��p�@p�>W
=@�C����p�@�\?�33AYG�Cff                                    Bxod;�  �          @�����@!녾�=q�'�C�R��@?�G�A�C�                                    BxodJ~  �          @�ff����@&ff�O\)���
C�=����@*=q?z�@��\C@                                     BxodY$  �          @��R���
@�ÿn{�Q�C  ���
@"�\>�p�@eC��                                    Bxodg�  �          @������?�G�@>{A�{C�=����=u@\��B\)C3Q�                                    Bxodvp  �          @�����{?���@w
=B�C#���{�=p�@���B'33C=}q                                    Bxod�  �          @\���\?�{@Z�HB	�C$G����\��@h��B�C:)                                    Bxod��  �          @������?���?�A�(�C ����?0��@!G�A�{C,��                                    Bxod�b  �          @������?�@I��A�G�C � ���ý�@dz�B�
C5n                                    Bxod�  �          @�Q����
?У�@)��A�  C"Y����
=�Q�@G
=A�{C2�                                    Bxod��  �          @Å��=q@�>B�\?�\C����=q?��?��RA@  C"#�                                    Bxod�T  �          @��
���
@��>k�@(�C!H���
?޸R?��\AB�\C"�                                    Bxod��  T          @Å��{@)����\)�)��C���{@(�?�=qA$(�C�f                                    Bxod�  
�          @��H����@@�׾����ffCG�����@5?��A%��C��                                    Bxod�F  �          @�����\@�>��?��C�����\?�33?�G�AAC!5�                                    Bxoe�  �          @����ff@ �׾�  �Q�C �{��ff?�\)?J=q@�\)C!�H                                    Bxoe�  �          @�=q��@�
�\�g�C ���?�(�?333@��
C �)                                    Bxoe&8  �          @��H��G�@=q>L��?�Cz���G�?�p�?���AS�C J=                                    Bxoe4�  �          @�(�����@�R?@  @��C������?�\?��A��C"L�                                    BxoeC�  �          @�����p�@>�(�@\)C����p�?�?�ffAh��C"G�                                    BxoeR*  �          @����\)@��>�@���C  ��\)?�?�  AaC#��                                    Bxoe`�  �          @������@�?
=@�G�C�����?��H?�
=A~=qC#&f                                    Bxoeov  �          @�����{@��?=p�@�ffC�
��{?��
?�  A�ffC$�3                                    Bxoe~  �          @�(���\)@�?E�@�C ����\)?�{?�Q�A�  C&�)                                    Bxoe��  T          @��
���R?��R?n{Az�C ����R?�  ?�A��C'��                                    Bxoe�h  �          @������
@�R?�\)A(Q�C\)���
?�{@A�G�C&h�                                    Bxoe�  �          @�(�����@G�?�{AL��C�3����?��\@z�A��C'�                                    Bxoe��  T          @\��{@�R?�{A'�Cu���{?˅@p�A�ffC#�                                     Bxoe�Z  �          @�33��
=@��?�A0��C�f��
=?\@  A��RC$xR                                    Bxoe�   �          @Å���
@(Q�?��RA;�C�H���
?��@��A�\)C#                                      Bxoe�  T          @��
��{@<(�?�\)AN�\C����{?�@*�HAΏ\C �                                     Bxoe�L  �          @�33��z�@8Q�?��Ak
=C�q��z�?ٙ�@2�\A�(�C!�3                                    Bxof�  �          @Å���R@8��?��
AB{C����R?�@$z�AǮC �                                     Bxof�  �          @Å��p�@'
=?�Q�A�=qC0���p�?��\@@  A�33C&@                                     Bxof>  �          @�p���G�@��?��RA��\C.��G�?���@=p�A�\)C(E                                    Bxof-�  �          @�����\@?��RA�33CT{���\?�  @9��A�\)C)c�                                    Bxof<�  �          @�(����@��?��HA�{C�H���?�Q�@,��A��C'p�                                    BxofK0  �          @���z�@(�?�\A��C����z�?���@0��Aԏ\C'�                                     BxofY�  �          @����p�@p�?��A�{C�
��p�?s33@/\)AӮC*�                                    Bxofh|  �          @�(���\)@p�?���Ar{C
��\)?�=q@   A��RC(�
                                    Bxofw"  
�          @�z���G�@p�?�{AMp�C=q��G�?��H@33A�z�C'�3                                    Bxof��  T          @�{���@G�?�G�A{C(����?�Q�@�A���C%                                    Bxof�n  �          @ƸR��@  ?���A#�Ch���?�\)@ffA��
C&p�                                    Bxof�  �          @ƸR����@��?��A@Q�C�3����?�p�@\)A��C'��                                    Bxof��  �          @�\)����@��?�ffA@��C&f����?��
@�A���C':�                                    Bxof�`  �          @�
=���
@�?�=qAD��C}q���
?���@�A�  C&�                                    Bxof�  �          @�����@ff?�Q�AX  C�����?��
@(�A��\C&�                                    Bxofݬ  �          @�����ff@��?�G�Ac�CJ=��ff?��
@!G�A�{C&Ǯ                                    Bxof�R  �          @��
���@?�{At��C�����?�
=@%�A�  C'�R                                    Bxof��  �          @Å��33@p�?�{At��CL���33?��@)��A�p�C&}q                                    Bxog	�  T          @\���
@�?�  Ae�C�����
?�p�@�RA��C'{                                    BxogD  T          @�\)����@=q?���AV�RC}q����?���@�A��C%�H                                    Bxog&�  "          @�����p�?��;�z��0  C!�f��p�?޸R?5@��C"��                                    Bxog5�  �          @��R��?�33������\C#����?�(�>���@L(�C#�                                    BxogD6  �          @����H?��
>��@$z�C"L����H?��?�33A4  C&�                                    BxogR�  �          @��R���\?�33?J=q@���C!+����\?�(�?�z�A���C'�f                                    Bxoga�  �          @��\��=q?�G�?�  Ap(�C!����=q?:�H@��A�  C,#�                                    Bxogp(  �          @�
=���R?��?�\)A�{C!�q���R?\)@�RA���C-�                                    Bxog~�  �          @�
=����?�z�?���Ai�C!�q����?5@�\A���C+�R                                    Bxog�t  �          @�����Q�?��
?ٙ�A�(�C"�q��Q�>�(�@  A���C/
                                    Bxog�  �          @������
?��?�{Ad��C#:����
?(�?��HA���C-&f                                    Bxog��  �          @������\?aG�?�(�A33C*5����\�#�
?��HA�33C4&f                                    Bxog�f  �          @�
=��
=?�\)?�ffA�  C!�q��
=?z�@
=qA��HC-W
                                    Bxog�  T          @��\���R@
�H?��A]�Cn���R?�z�@G�A�C&�)                                    Bxogֲ  �          @������?�33?�A��C$n����>L��@�Aə�C1�3                                    Bxog�X  �          @�
=��z�?��?�\)A�  C�=��z�?G�@�A�  C+n                                    Bxog��  �          @�{���@
=?�=qAX��C�����?�{@\)A�33C'��                                    Bxoh�  �          @������@	��?���A�{CY�����?c�
@*�HA�\)C)�R                                    BxohJ  �          @����@=q>�{@[�C
=��?��?�ffAz�RC 
=                                    Bxoh�  �          @�G�����@(Q�8Q��=qC������@?��HAA�C�\                                    Bxoh.�  �          @�����33@Q�>aG�@	��CJ=��33?��H?��AO�C"O\                                    Bxoh=<  �          @��
���?�
=>��@���C%�����?�G�?��A-p�C)��                                    BxohK�  �          @������?(�?Y��A��C.  ����=��
?��A#\)C3=q                                    BxohZ�  T          @����p�?\?\)@�ffC$�3��p�?�G�?��\AG\)C)�                                    Bxohi.  �          @�{���
?���>�  @�HC!�����
?���?�
=A8  C%��                                    Bxohw�  �          @�����
?��
>B�\?�ffC"s3���
?�?��A)C%�{                                    Bxoh�z  
�          @����  @Q�����{�C޸��  @G�?E�@�{C�=                                    Bxoh�   �          @�(���Q�@!녿+����CJ=��Q�@ ��?@  @��HC}q                                    Bxoh��  �          @����33@�ÿTz���C&f��33@��>��@���C\                                    Bxoh�l  �          @�=q��{@$z��(����C����{@=q?z�HA
=C{                                    Bxoh�  �          @�33���H@0  �#�
����C�����H@+�?fffAffC:�                                    Bxohϸ  �          @�33���\@!녿����V�\C����\@7�>.{?ٙ�C�{                                    Bxoh�^  �          @�=q����@�ÿ���5p�C�����@(Q�>�\)@3�
C�                                    Bxoh�  (          @�����(�?�z�aG��z�C u���(�@>�  @\)C�=                                    Bxoh��  �          @��H��G�?�{�E����HC#�\��G�?�\>B�\?�\)C"@                                     Bxoi
P  �          @������\?�=q�����C&�=���\?�z�>�  @!G�C%�=                                    Bxoi�  �          @�  ����?�Q�=#�
>�
=C%s3����?���?J=q@�ffC'                                    Bxoi'�  "          @�������?�z�=�G�?��C ������?Ǯ?���A0��C#޸                                    Bxoi6B  "          @�  ��  @�ÿ�p��o�C� ��  @3�
�#�
�uC�                                    BxoiD�  �          @����{>aG�@G
=B��C1T{��{����@0��A�\CE+�                                    BxoiS�  
�          @�ff��=q?���=�?�p�Cٚ��=q?˅?���A8(�C#L�                                    Bxoib4  
�          @�{���?�{>\@tz�C(�3���?=p�?h��A
=C,^�                                    Bxoip�  T          @�������?�\)>��
@Mp�C#� ����?���?���A3�C'                                    Bxoi�  
Z          @������H?�=q>��
@N{C&�)���H?u?z�HA(�C*@                                     Bxoi�&  �          @�Q�����?��?=p�@�{C&������?333?��AS33C,�                                    Bxoi��  �          @������?p��?�=qA,��C*� ���>k�?�AbffC1�3                                    Bxoi�r  
�          @�����=q?�ff>��@��
C&����=q?aG�?��
A&=qC*��                                    Bxoi�  �          @����G�?�?Tz�A��C(\��G�?\)?���AU��C.@                                     BxoiȾ  �          @�Q����\?���?=p�@��
C(�����\?�?�(�AD��C.!H                                    Bxoi�d  �          @�\)��z�?z�H�u��RC*(���z�?\(�>��@��C+L�                                    Bxoi�
  �          @�\)��z�?}p���\)�.{C*���z�?^�R>�@���C+0�                                    Bxoi��  �          @�Q�����?k������\C*�����?��<�>���C)\)                                    BxojV  
�          @�\)���?Q녾k���C+� ���?Q�>u@=qC+�=                                    Bxoj�  �          @�
=��33?�  ��
=���RC)����33?���>.{?޸RC)33                                    Bxoj �  
�          @�{����?z�H�E���=qC)�R����?��R�\)����C'aH                                    Bxoj/H  �          @����(�?.{�Y���	�C-&f��(�?��
��p��j=qC)��                                    Bxoj=�  
(          @�\)��ff>�
=��=q�.�RC/�\��ff?   �#�
����C/                                      BxojL�  �          @�Q���\)>�z��ff��  C1
��\)>��H�k��G�C/&f                                    Bxoj[:  T          @�G�����=�Q�>u@ffC3{���ýL��>�  @�RC4xR                                    Bxoji�  "          @����  ��\)?@  @�C4����  ��?��@�C8�=                                    Bxojx�  �          @�����p��#�
?��A.�RC5�H��p��8Q�?W
=A�C;8R                                    Bxoj�,  �          @��\���;8Q�?�{AX(�C5�����Ϳ^�R?��A'\)C<�                                    Bxoj��  
�          @��H��z�#�
?��HAg�C5�{��z�h��?�33A6ffC=!H                                    Bxoj�x  �          @��\���Ϳ�\?�(�AB{C9����Ϳ�=q?B�\@��C>�=                                    Bxoj�  "          @��\��{��ff?(�@�
=C>s3��{���H�L�;�ffC@�                                    Bxoj��  
�          @�=q��\)�B�\>�(�@�\)C;����\)�^�R�#�
���C<��                                    Bxoj�j  
�          @�����?h��?���A�p�C*�f���þ�=q@Q�A���C6�                                    Bxoj�  T          @��R���\?�Q�@��A�33C'k����\��=q@*=qA�C6޸                                    Bxoj��  "          @��R��G�?�  @#33AʸRC)T{��G���@,(�A֣�C9�\                                    Bxoj�\  
�          @�����R?�=q?�
=A��C(�=���R��G�@p�A�Q�C50�                                    Bxok  �          @��\���?�ff?O\)A��C&ٚ���?+�?���A]p�C-+�                                    Bxok�  T          @�=q���H?��þ�G���G�C$E���H?Ǯ>��@�z�C$^�                                    Bxok(N  �          @�����
=?�(��#�
���C"����
=?��>�p�@k�C!ٚ                                    Bxok6�  �          @�G���
=?��þ�ff��
=C!����
=?��
?
=@��
C!��                                    BxokE�  �          @�����G�@(��B�\��C����G�@��?�@�C�f                                    BxokT@  �          @������
?��aG��
�HC!
���
?�?L��A��C"�R                                    Bxokb�  �          @�ff��33?�\)?�Q�A`  C&B���33>Ǯ?���A�
=C0�                                    Bxokq�  �          @��R���\?��?��HA<z�C#�����\?=p�?�33A�\)C,xR                                    Bxok�2  "          @�ff����?���?�(�A=�C&B�����?�\?�\A��RC.�\                                    Bxok��  T          @�
=����?��?�(�Ac�
C#xR����?��@�A��\C-                                    Bxok�~  �          @��R���\?�p�?��A0��C"� ���\?Y��?��A�=qC+O\                                    Bxok�$  "          @�����?�=q?}p�Ap�C!Ǯ����?�G�?�A�ffC)��                                    Bxok��  �          @�(���\)?�z�?k�AQ�C �\��\)?�\)?�ffA���C(xR                                    Bxok�p  �          @��H���R?�
=?�z�A7
=C"�H���R?L��?��A�Q�C+�f                                    Bxok�  "          @��
���@�?�{AUC�����?���@
=A�{C'#�                                    Bxok�  T          @�����\?�?�A`��C +����\?^�R@�RA�33C*�=                                    Bxok�b  "          @�(�����@
=q?��AZ�RC������?�=q@�A��C(ff                                    Bxol  �          @��H��Q�@p�?�A8��C5���Q�?��R@
�HA�
=C&��                                    Bxol�  T          @��\��Q�@?��A]p�C\)��Q�?�G�@33A���C)
                                    Bxol!T  �          @�33��z�?�z�?�Q�A;\)C }q��z�?z�H@G�A�=qC)��                                    Bxol/�  "          @�����R@  ?���AV=qC�����R?�
=@ffA��C'G�                                    Bxol>�  T          @�z���Q�?�  ?Tz�A33C"\)��Q�?��
?��A��RC)c�                                    BxolMF  T          @�(����R?���>��@"�\C&p����R?��\?uA=qC)ٚ                                    Bxol[�  �          @���\)?�
=>��@�  C%�3��\)?}p�?�\)A.=qC*5�                                    Bxolj�  �          @��R����?���>�33@Z�HC&�f����?p��?��\A=qC*�R                                    Bxoly8  
�          @����Q�?��>aG�@�C'+���Q�?}p�?c�
A
ffC*@                                     Bxol��  T          @�z���
=?��>aG�@
=qC&�=��
=?�G�?h��A{C)��                                    Bxol��  
�          @����\)?��>aG�@ffC&ff��\)?�ff?n{A��C)�
                                    Bxol�*  "          @�{��ff?��?�@�p�C$ٚ��ff?�G�?��
AG�C)��                                    Bxol��  T          @��R���H?���?�Q�A9C$O\���H?.{?���A�p�C-
=                                    Bxol�v  �          @�Q���?�=q?�
=A[�C&޸��>�33?�A��RC0xR                                    Bxol�  �          @�
=���
?\?��AH  C$�����
?
=?�z�A�G�C.                                      Bxol��  �          @��R���?�Q�?�  AAp�C �����?u@A���C*�                                    Bxol�h  �          @�
=���?�
=?��APQ�C#(����?333@�\A��C,�)                                    Bxol�  T          @���z�?�
=?���A��C"����z�>�(�@(�A�{C/k�                                    Bxom�  �          @�ff��  ?�\?�{AT(�C"(���  ?B�\@ffA��
C,#�                                    BxomZ  
�          @�ff��?޸R?�{A{�C"@ ��?��@�\A��
C-�                                    Bxom)   
�          @�ff����?�=q?�(�Ad��C$  ����?
=q@A���C.\)                                    Bxom7�  "          @�p����@�H?��ArffC�f���?��H@&ffA�33C&�
                                    BxomFL  �          @�33��=q@p�?�(�A�{Cs3��=q?p��@(��A���C)�=                                    BxomT�  �          @�\)���þ8Q�@�A��C5�����ÿ�  ?�{A�(�CAG�                                    Bxomc�  �          @������
�+�@�A�=qC;k����
���?ǮA�(�CG@                                     Bxomr>  
�          @�����(�=�\)?ٙ�A�ffC3G���(��\(�?�(�Aq��C={                                    Bxom��  
�          @�p����=#�
?�z�A�(�C3�=����\(�?�AhQ�C<�q                                    Bxom��  "          @��R��p�>\)?�\)A���C2�=��p��B�\?�Q�Ak\)C;�                                    Bxom�0  
Z          @�  ����Tz�@(�A�33C<�R������?�33Aa�CG��                                    Bxom��  
�          @��R��ff�h��?�z�Ad��C=xR��ff�\?0��@�  CC�H                                    Bxom�|  
�          @�{���\��\)?�Q�Am��C?�
���\��(�?(�@ǮCE�H                                    Bxom�"  	�          @��
��  �\(�?&ff@�{C<�H��  ��=q=�Q�?aG�C?�                                    Bxom��  �          @�{��{�p�?:�H@�p�CL&f��{��׿z��ÅCL��                                    Bxom�n  �          @�  ��  ���>��@�G�CN���  �G��fff��HCM��                                    Bxom�  �          @�������u�(�CA�f���(�����{
=C:n                                    Bxon�  �          @�33������
�(����\)CAG�����8Q쿞�R�L��C;�
                                    Bxon`  �          @��H��  ��\)�z�H�!��CE0���  �W
=�ٙ����C={                                    Bxon"  �          @��R��Q�� �׿L���33CI�)��Q쿡G���G���z�CB�                                    Bxon0�  �          @�33������G��.�\CKu���녿��H��p����CB#�                                    Bxon?R  �          @�G���=q��ͿW
=��\CP8R��=q��{������CGp�                                    BxonM�  
�          @�{��33�Ǯ�z�H�%�CE���33�J=q��z�����C<                                    Bxon\�  �          @�����\)?L�Ϳ�
=�H��C+L���\)?���\)��p�C%�                                    BxonkD  �          @�����p�@H�ÿ�\)�;�C����p�@P  ?@  @�Q�C��                                    Bxony�  �          @��
���@K�����333C� ���@P��?L��A�HC�                                    Bxon��  �          @�(����@0�׿�G���=qC�
���@QG�<#�
=��
CW
                                    Bxon�6  �          @�z����\@3�
��33���\CxR���\@P  >�?���C�
                                    Bxon��  �          @�������@*=q��33���C������@HQ�=L��?   CO\                                    Bxon��  �          @�33����@'
=����<��C}q����@3�
>�ff@��C��                                    Bxon�(  �          @�(����
@�\���R�x  C����
@/\)���
�.{C�f                                    Bxon��  �          @�=q���R@(���G����
C
���R@2�\�����H��CW
                                    Bxon�t  �          @������@&ff���\�)G�C�����@/\)?��@���CO\                                    Bxon�  �          @������@>�R��
=��{C������@c33���
�L��C
                                    Bxon��  �          @��R��G�@S�
�����  CaH��G�@p  >�\)@6ffC
�)                                    Bxoof  T          @�(��|��@G��\)�ң�C��|��@~{�����C�
                                    Bxoo  �          @�
=�2�\@<(��i���(33C}q�2�\@�{���R�}�B�(�                                    Bxoo)�  �          @����.�R@c33�N{��RB�(��.�R@�
=�J=q�\)B�\                                    Bxoo8X  �          @��R�#�
@O\)�a��"33C ^��#�
@�(���(��P(�B��f                                    BxooF�  �          @�{�U�@2�\�>{�	ffC\�U�@|�Ϳ}p��-��C#�                                    BxooU�  �          @���y��@-p��!G���C5��y��@h�ÿ(�����C��                                    BxoodJ  �          @�p���Q�@>{���H�z�\C���Q�@R�\>\@��CO\                                    Bxoor�  �          @�(��*=q@XQ��C33�(�C .�*=q@���=p�� (�B�G�                                    Bxoo��  �          @���=q@mp��e�&�HB�{��=q@�녿��\�.{B֞�                                    Bxoo�<  �          @��׿�(�@vff�Z=q�
=B�LͿ�(�@��\�J=q�  B�u�                                    Bxoo��  �          @�Q��(�@tz��R�\���B�q�(�@���333���HB�u�                                    Bxoo��  �          @�G��Q�@Z�H��p��GG�B��ͿQ�@�(���z���p�B�\                                    Bxoo�.  �          @��ÿ�@hQ���33�@��B�uÿ�@��׿��R�z�HB�aH                                    Bxoo��  
�          @�p�>�\)@�ff�QG��Q�B�aH>�\)@�������B���                                    Bxoo�z  �          @����z�@z=q�j=q�#33B��ÿ�z�@�Q�xQ���B��                                    Bxoo�   �          @��
��=q@p���u�-
=Bᙚ��=q@�������EBՔ{                                    Bxoo��  �          @����Vff?�33?���A�  C���Vff?+�@   A�
=C(��                                    Bxopl  �          @�  �]p�?�ff@eB-��C�\�]p��!G�@w
=B>�
C>h�                                    Bxop  �          @�����R@1G�?��HA��
C����R?�z�@<(�B��C!�                                    Bxop"�  �          @��H�~�R@Vff?�ffA6�\C�3�~�R@  @,(�A�C��                                    Bxop1^  �          @����QG�@��ÿY�����C)�QG�@vff?��HA���Cc�                                    Bxop@  �          @�ff�Z�H@k��z�H�0Q�C�f�Z�H@hQ�?�z�ARffCQ�                                    BxopN�  �          @�(��HQ�@^{�'���  C\�HQ�@��H�����aG�B��3                                    Bxop]P  �          @�p��7
=@L(��L����C�H�7
=@�p��xQ��'
=B��                                    Bxopk�  �          @���Z�H@^{�(����C���Z�H@�33��33�g
=C +�                                    Bxopz�  �          @��\�J�H@_\)�>{� �HCB��J�H@�G���R�˅B��
                                    Bxop�B  
�          @��\�:=q@[��6ff�z�CT{�:=q@�p������p�B��                                    Bxop��  �          @���@  @W��>{�33C���@  @�{�.{��RB�8R                                    Bxop��  �          @���-p�@QG��U��RC���-p�@�녿�ff�2�HB�                                    Bxop�4  �          @��
�5�@E��W��ffC�{�5�@�����J{B�aH                                    Bxop��  �          @�{�333@0���n{�-�RCaH�333@��H�����\)B��3                                    BxopҀ  �          @��R���@e��U��B������@�녿aG����B��                                    Bxop�&  O          @���(�@Y���^�R�"�RB��\�(�@������=G�B噚                                    Bxop��  Y          @�=q�ff@0���x���<G�C^��ff@�p������\)B��)                                    Bxop�r  "          @�
=�g
=@;��������C�H�g
=@aG��u�+�C��                                    Bxoq  T          @�
=��{@*=q��\����C\��{@!G�?�  A1�Cp�                                    Bxoq�  T          @����~{@O\)���\�4(�C�=�~{@Q�?c�
AQ�Cs3                                    Bxoq*d            @����r�\@W��8Q�� z�C
J=�r�\@N�R?�(�A[�
C��                                    Bxoq9
  Y          @���k�@]p��Q���HC�=�k�@Vff?�
=AT��C	��                                    BxoqG�  �          @�33�L(�@s�
�����  C�R�L(�@�\)>��@���B�.                                    BxoqVV  
�          @�p��>�R@\(��AG��33C�H�>�R@��׿0����  B��R                                    Bxoqd�  T          @�����\@��H�'���B� ��\@��    �L��B�z�                                    Bxoqs�  T          @��
���@��\�����\)B��)���@�Q�>�?�33B�u�                                    Bxoq�H  "          @�(����R@��H�2�\��=qB��H���R@�ff�������B���                                    Bxoq��  T          @��R�,(�@n{����z�B����,(�@��    =L��B���                                    Bxoq��  �          @�(��G�@c33�C�
�z�B�W
�G�@�z�+���B�{                                    Bxoq�:  
�          @�  ��G�@33?^�RA%p�C���G�?�G�?�=qA�(�C#�                                    Bxoq��  "          @�z��B�\@i���
=q�ϮC�B�\@W�?�G�A�G�C�                                    Bxoqˆ  "          @����x��@!�?���A[�C��x��?\@z�A�=qC�{                                    Bxoq�,  �          @����\��@N�R?z�@޸RC�)�\��@=q@{A��HC                                      Bxoq��  �          @����Tz�@XQ�>#�
?���C�\�Tz�@0��?���A�z�CO\                                    Bxoq�x  
�          @�Q��_\)@ff?�  A�(�C  �_\)?�ff@/\)B�C#Q�                                    Bxor  "          @�G��Tz�@$z�@
=A��
C8R�Tz�?��
@J�HB)Q�C"��                                    Bxor�  �          @��
�c33@p���Q����CG��c33@6ff=�G�?�33CG�                                    Bxor#j  �          @�G��`��?n{�C�
� p�C%!H�`��@�����CO\                                    Bxor2  
�          @���fff?��
�O\)�5C���fff?�Q�>�\)@x��C�3                                    Bxor@�  T          @���)��@#�
�	����{C�)��@S�
��G����C ��                                    BxorO\  �          @���\)@33�����v{B��=��\)@��/\)��G�B�ff                                    Bxor^  �          @��
�\��@��E���CxR�\��@\�Ϳ�(����\C�                                    Bxorl�  �          @�ff�XQ�?�ff�\(��'�\C��XQ�@W
=��Q�����C33                                    Bxor{N  T          @�
=�R�\?�33�u��>C ��R�\@A��'���ffC	^�                                    Bxor��  �          @�{�Q�?�
=�p  �9p�C��Q�@Mp��������C�{                                    Bxor��  
Z          @�  �QG�?L���~�R�G(�C&5��QG�@3�
�;��CL�                                    