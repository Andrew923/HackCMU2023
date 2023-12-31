CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230223000000_e20230223235959_p20230224021612_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-24T02:16:12.511Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-23T00:00:00.000Z   time_coverage_end         2023-02-23T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill            records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxg�@  
Z          A"�\��(�@���c33��\)C
xR��(�@�>#�
?aG�C�\                                    Bxg�+�  "          A �����@љ��G����C�����@�
=?#�
@j=qC�q                                    Bxg�:�  T          AG���p�@�z���\�?�C����p�@�?�\)A/�
C��                                    Bxg�I2  �          A��Q�@<�������C����Q�@�ff�%���\C��                                    Bxg�W�  q          A ������@(������
=C"�\����@���aG����C�                                    Bxg�f~  
o          A
=�{@z=q�aG���Q�CL��{@�������C��                                    Bxg�u$  �          A  ��R@C�
��G��хCp���R@�=q����K�CxR                                    Bxg���  "          A
=���?�\��33�Q�C'Q����@���mp�����C��                                    Bxg��p  
�          A�H���H?��������C'�����H@��s�
��C�                                    Bxg��  T          A��G�@ff��z����C%�
�G�@z=q�   �s�Cٚ                                    Bxg���            A�����?�z��~{��  C'���@j=q����f�\Cs3                                    Bxg��b  
�          A=q���?�����H�ÅC'�f���@l���$z��p��C33                                    Bxg��  �          A{�33@G��mp���(�C'E�33@g����K33C��                                    Bxg�ۮ  
�          Aff��?�z��A����HC(L���@K���33���C ��                                    Bxg��T  �          A��H?����  �TQ�C*�=��H@   ��33�ָRC%+�                                    Bxg���  T          A��
?�  ��
�X��C,xR��
@  ��������C&��                                    Bxg��  �          A=q�  >��H�&ff�t��C1\�  ?޸R�   �;33C)�H                                    Bxg�F  
Z          A�H�33?
=��
�?\)C0�=�33?�ff��p��	�C*�                                    Bxg�$�  �          A{�
=?\(����H�
=C.��
=?�{�������C*�
                                    Bxg�3�  �          A  �G�?��ÿ��C-���G�?�녿.{����C*O\                                    Bxg�B8  �          A����?W
=������=qC.�����?�33�8Q���
=C+�R                                    Bxg�P�  �          A��{?
=��z����HC0xR�{?��Ϳ333���
C-��                                    Bxg�_�  T          A���>Ǯ�����{C1����?�G��u��{C-�q                                    Bxg�n*  T          Aff���>L�Ϳ�{�p�C2�����?Y��������z�C.��                                    Bxg�|�  �          Ap��=q�O\)��p���RC8���=q=L�Ϳ�Q��"�RC3��                                    Bxg��v  "          AQ���H��
=�33�F�RC<�H��H�W
=�\)�r�RC5J=                                    Bxg��  T          A  �
{��
=�Z=q��z�C@�
�
{=L���z=q���C3�f                                    Bxg���  �          A��33��ff�c�
����C?��33>u�~�R����C2h�                                    Bxg��h  �          Ap��	p���G��x����33C?���	p�>�G������  C1\                                    Bxg��  
�          A���
=����}p�����C@���
=>�Q����
���HC1��                                    Bxg�Դ  "          A(���h���Y����\)CL+����
=�������\C>�                                    Bxg��Z  
�          A�
��R�mp��G�����CLh���R�У�������
C?L�                                    Bxg��   
�          A�
��
�p  �1G���=qCL� ��
����������G�C@��                                    Bxg� �  
          A33�33�}p��!��x(�CM���33�
=q�����
CB��                                    Bxg�L  
�          A
=��H��z��{�Y��CN޸��H�{��  �ȸRCD�=                                    Bxg��  �          A(�����33�>�R��(�CQ�q�������Q����CD�\                                    Bxg�,�  "          A�������33�Y�����CU
����G���  ��CE�
                                    Bxg�;>  �          A33��\)����*=q��z�CT:���\)�&ff���H���HCG�\                                    Bxg�I�  �          A��������z��?\)��(�CU�f����� ������{CH                                      Bxg�X�  "          A�������  ���
��=qC\8R��������=q�/G�CH��                                    Bxg�g0  �          A����������n{��Q�C[33�����33��(��&�\CIJ=                                    Bxg�u�  "          A	���=q�s�
�C�
����CPL���=q��G��������CB�                                    Bxg��|  �          A	����=q��33�u���Q�C\�{��=q��
����2�CI33                                    Bxg��"  �          A33���H��ff��G��3�HCz:���H��������)C^�                                    Bxg���  T          A�
��
=��33��{�A�C�P���
=��(���(�Ck�
                                    Bxg��n  �          A�
�����ȣ���{�<
=C�R���ÿ��H�=qG�Cgff                                    Bxg��  �          A���\)�ȣ������6(�CzL��\)��\�(���C^=q                                    Bxg�ͺ  �          A"�\�\)������\)�G�\Cw���\)�����\)z�CP��                                    Bxg��`  T          A�\��  ������p���G�Ci=q��  �AG����H�N\)CU��                                    Bxg��  �          A��������@p�AW�CZ������Å�xQ�����C\xR                                    Bxg���  
�          A\)��  ��{?n{@���C`� ��  �����=q�s33C^�                                    Bxg�R  
�          Az����R�љ�?�(�A=qCc�����R����   �Mp�Cc\                                    Bxg��  T          AG���(���z�@#�
A��Ce����(����ÿ������CgW
                                    Bxg�%�  
�          A�������\)@X��A���CdL������33�#�
���Cg�                                    Bxg�4D  �          A  ��G���@��
A�\)Cc)��G���Q�?�ffA	�CjB�                                    Bxg�B�  
�          A������Q�@�\AW
=Ch������(������(Q�Ci                                    Bxg�Q�  "          A{��p���Q�?O\)@�\)Cf���p��ȣ��*=q���Cc�f                                    Bxg�`6  
�          A  ������=��
?
=qCc�3�����=q�O\)���\C_�
                                    Bxg�n�  T          A����{��ff?�
=A�RCe� ��{��  �p��\��Cd�f                                    Bxg�}�  �          A=q��ff���@	��AY�Ch&f��ff�����\)�"�RCh��                                    Bxg��(  
F          A33��{��33?��HA�RCe���{���Q��VffCd\)                                    Bxg���  
�          A��������@	��AW33CeG������\)��G���\Ce�R                                    Bxg��t  �          A���=q���H@7�A�33Cc����=q�ָR���_\)CfW
                                    Bxg��  
Z          A���������@6ffA��Ca33���������W
=Cc��                                    Bxg���  
�          A\)�����=q@8��A�z�Ca�=����ָR�   �L(�Cd��                                    Bxg��f  �          A�����\)@,��A�(�Ca�������  �:�H��(�Cd
                                    Bxg��  
�          A�������(�@>�RA�z�C\��������
�B�\��(�C`G�                                    Bxg��  �          A
�\������@]p�A�=qC\�3����>��@G
=Cb33                                    Bxg�X  T          A�����R��p�@g
=A�33CcT{���R��>W
=?��Cg�=                                    Bxg��  "          A�
��=q��Q�@h��A˅Ca����=q���H>�ff@B�\Cg�                                    Bxg��  
�          A
=�����(�@��HA�{Ca�������=q?�\)@��Ch�                                     Bxg�-J  "          A
�\��{��  @�ffA���C^����{�љ�?�z�A�\Cf@                                     Bxg�;�  �          A�������H@�(�B&�CZ=q����ff@FffA�Q�Cg5�                                    Bxg�J�  "          A(���z��h��@�
=B6Q�CX����z���\)@g�A�z�Cg��                                    Bxg�Y<  �          A(��y������@�
=B9�\Cb��y���љ�@J=qA�=qCo=q                                    Bxg�g�  �          @���k���G�?�\A_�C�K��k���  ���s�
C�Ff                                    Bxg�v�  "          @�(�?�����{�W
=��  C�  ?�����Q��U��\)C�E                                    Bxg��.  
�          @��H>\)�Mp�@�  BS�HC�8R>\)��(�@Q�A���C��                                    Bxg���  	�          A �׿�p���\)@���B��C^�R��p���Q�@ÅBE�
C}��                                    Bxg��z  T          AG��녿���A
=B���CVW
����33@�BECy@                                     Bxg��   �          A���Q쿈��A�
B��CTE��Q���
=@��BN��C{0�                                    Bxg���  T          @�������@�B�ǮCVǮ�����
=@�Q�B-��Cs
                                    Bxg��l  
�          @����5��c�
@ҏ\B�u�CEn�5���  @��B=�Cj�                                    Bxg��  
�          @�33�%�(�@�33B�33CA=q�%��(�@ÅBMCm�H                                    Bxg��  	�          @��\�&ff>\@��
B�ǮC+��&ff�QG�@��
Bf�Cg�=                                    Bxg��^  
�          @�G��=q����@陚B�33CK��=q��  @��HBCp�CqǮ                                    Bxg�	  �          @����녿(��@�z�B�k�CD)����@�(�BP��Cqk�                                    Bxg��  
�          @�p��#�
?�@�z�B��)C(Y��#�
�AG�@ϮBj�CeǮ                                    Bxg�&P  �          A	� ���)��@��B��=Cb��� �����H@�
=B�
Cw��                                    Bxg�4�  �          A(���H�QG�@��Bq��Ci� ��H���H@��B	�HCy�                                    Bxg�C�  
�          A��E��@�ffBx��CZ&f�E��  @��\B��Cq�q                                    Bxg�RB  
�          AQ��p���W
=@ۅBV�RC]���p���ȣ�@��A�ffCo�                                    Bxg�`�  �          A�l(���(�@��BE��Cgs3�l(�����@r�\A���Cs��                                    Bxg�o�  �          A(��1����H@��B7\)Ct�{�1��p�@?\)A�{C{�
                                    Bxg�~4  T          A\)�.{��Q�@�\)B��Cu��.{��G�?�z�A5��Cz�=                                    Bxg���  T          @\��R�tz�@��BB��C�]q��R��\)@   A��C��H                                    Bxg���  
�          @ҏ\�����@W
=A���C�LͿ����(�>��R@0  C�E                                    Bxg��&  
�          @����(Q����H�{����Cqs3�(Q��J=q�����8�RCf0�                                    Bxg���  �          @��R�z��y���5� �Co.�z���\���
�Z�HC]8R                                    Bxg��r  
�          @�G���Q��:�H?ٙ�A�Q�C�t{��Q��XQ�=��
?���C��                                    Bxg��  	t          @�\)?��R�Y��@���B�.C�33?��R��  @���BV  C�(�                                    Bxg��  "          @�(�?���?z�@�\B�L�A��?����I��@�p�Bx�RC�8R                                    Bxg��d  
�          @�@�@	��@�p�B�G�B-�\@녿���@�B�\)C���                                    Bxg�
  
�          @�  @ff@?\)@��
Bd\)BOQ�@ff�W
=@ҏ\B���C�l�                                    Bxg��  �          @�=q@*=q@�@��
Bp�HB"{@*=q�s33@θRB��RC�'�                                    Bxg�V  
�          A
=@p���(��@��
BhG�C�w
@p�����@�{B�
C�XR                                    Bxg�-�  �          A
{@���G�@��
BYffC��@����=q@���B�
C��f                                    Bxg�<�  �          @�\)@�
=�(�@ҏ\B^Q�C��@�
=�l��@�\)B-p�C�/\                                    Bxg�KH  
�          A ��@n�R���
@�{Bo�C��{@n�R��p�@��B'p�C�Ff                                    Bxg�Y�  �          Ap�@j=q�>�R@��
B^
=C�j=@j=q���@�33B  C��H                                    Bxg�h�  �          A�@g��;�@޸RB`C���@g�����@�ffB�C���                                    Bxg�w:  T          A�@����b�\@���B@G�C��3@������
@tz�A�33C�ٚ                                    Bxg���  T          A=q@�p����@��A�p�C���@�p��\?�A�\C�޸                                    Bxg���  T          A�H@��R�|��@�(�B1{C�e@��R���@C33A�(�C�*=                                    Bxg��,  
�          @�@/\)����@��HBC��C�R@/\)�ƸR@?\)A�Q�C���                                    Bxg���  �          @ᙚ@B�\����@��B�C�*=@B�\��ff?�\)A333C��                                    Bxg��x  T          @�=q@Q���{@<(�A��C��)@Q����ͽ��}p�C��\                                    Bxg��  �          AQ�@����\@qG�A�
=C���@���\)��z��z�C��f                                    Bxg���  
�          A�@G����R@�\)A��C�8R@G���\>�=q?ٙ�C�(�                                    Bxg��j  "          AQ�@{���@��A�=qC��{@{�����L��C��                                    Bxg��  
�          A��?���(�@��HA�(�C��
?��{�\)�c�
C�R                                    Bxg�	�  
�          A��?��H��\@�\)A�Q�C�{?��H�33����fffC��                                    Bxg�\  
�          AQ�?�����@P��A�ffC��{?����׿�ff���
C���                                    Bxg�'  �          A�?�p����@S�
A��
C��)?�p���
=���j=qC��
                                    Bxg�5�  T          A(�?�Q��
=q?0��@���C���?�Q���{�^�R����C�E                                    Bxg�DN  �          A�?��\�=q=���?5C�z�?��\��{�w���=qC��=                                    Bxg�R�  �          @陚�c�
�&ff@ÅBm\)C>L��c�
�]p�@��\B6��C`!H                                    Bxg�a�  T          @�=q�U���\)@�{By{CF���U���z�@��
B5
=Cg8R                                    Bxg�p@  "          @��@  ��@�=qB���C=\�@  �w�@�\)BOQ�ChB�                                    Bxg�~�  �          A���Q���  @��HB(z�Cxٚ�Q����
@�Av�RC}�H                                    Bxg���  �          A�׿G��z�>�\)?�\)C�c׿G������qG��У�C�R                                    Bxg��2  �          @��׿Q���G�?��A�
C��=�Q���G������C��                                    Bxg���  
�          A\)���z��O\)��33C~G���������  �=  Cx�                                    Bxg��~  �          A���G���H�O\)��=qC�3�G���ff��ff�?��Cz��                                    Bxg��$  "          A���Q���p��b�\��(�C��׾�Q����
��\)�R�C��
                                    Bxg���  �          @��@  ��R��ff�^�RC�׿@  �������R�&��C�q                                    Bxg��p  
Z          @�Q�>���{��\��=qC���>���p��dz�� �C�S3                                    Bxg��  	`          @�\>W
=��R��p��{C��=>W
=�������\��C���                                    Bxg��  T          A�=��
�����p��C�C�H�=��
��Q���Q�� 33C�]q                                    Bxg�b  �          A�?   �  ��\�Ap�C��R?   ��
=��p��Q�C�0�                                    Bxg�   �          AG�?��R��\)��p��@z�C�L�?��R��Q���  ��HC��
                                    Bxg�.�  
�          A��?�  �  ��G��Q�C���?�  �ָR���R�33C�J=                                    Bxg�=T  "          A��?��
�=q���H�Q�C�*=?��
��ff���\���C���                                    Bxg�K�  �          A��?@  �\)��R��=qC�t{?@  ��p���33���C��                                     Bxg�Z�  �          Aff>����녿���vffC�N>�����\��z����C��f                                    Bxg�iF  T          A��>�
=�(�����3�
C�w
>�
=��(���{���C��3                                    Bxg�w�  
�          @陚��ff��������\CO� ��ff�u��=q�6C6�3                                    Bxg���  
�          @�\)��  @g�������
=C�f��  @�=q����bffC	Y�                                    Bxg��8  
x          A
=���@HQ����
�\C����@����;����
CǮ                                    Bxg���  h          A�����@�����\�Q�CE���@���I������Ch�                                    Bxg���  
�          @�����{@(Q������RC&f��{@�  ��R��Q�Ch�                                    Bxg��*  
�          A�H����@G��p  �ڣ�C!Ǯ����@p  ��\���
C�                                     Bxg���  
Z          A33�Ϯ��������C8�
�Ϯ?�\����	��C$��                                    Bxg��v  
�          A �����Ϳ!G����H�  C9c�����?�Q���z�����C'�\                                    Bxg��  
�          A{��ff���
��z���33C<+���ff?z�H�������C,:�                                    Bxg���  T          AQ������\�a��ˮCE�q��녾�
=��{��33C7^�                                    Bxg�
h  �          @����\)��(���  ��CX=q��\)�Tz��G���{CQ�                                    Bxg�  
�          @������H���>W
=?�
=Cf����H��\)�����CcO\                                    Bxg�'�  T          @�{��(����R�#�
��=qCh����(������(����33Ce0�                                    Bxg�6Z  
�          @ҏ\�p�����@@  A�z�Cvh��p����
>�Q�@S�
Cyc�                                    Bxg�E   
�          @ٙ���ff�}p�@���BKG�C0���ff��=q@*�HA�=qC��f                                    Bxg�S�  �          @�R����陚?!G�@�z�C�b������G��1G���\)C�B�                                    Bxg�bL  T          @��R    ��=q?��\A�C��    ��G�������C��                                    Bxg�p�  
Z          @��H>L�����?��RA6�\C�Ǯ>L���������y�C�˅                                    Bxg��  
�          @���?�(���{?���A\  C��?�(���
=�޸R�N�RC��H                                    Bxg>  "          @���?=p���  ?�{A]p�C��{?=p����ÿ�  �P  C���                                    Bxg�  
�          @�
=�dz���p��@  ���HCp��dz���(��e����
Clu�                                    Bxg«�  
�          @�  ��Q���{�������Ccu���Q��b�\����z�CX��                                    Bxgº0  �          @��Ϳk���G�?��
AQ�C��=�k���33������Q�C���                                    Bxg���  
�          @�33@*=q��Q�@���B;�C�Ǯ@*=q���@1�A�  C�!H                                    Bxg��|  
(          @�Q�@{��(�@c33B{C�j=@{��ff?xQ�A	�C�@                                     Bxg��"  �          @�?
=q��
=?��
AM�C�K�?
=q��{��z��]C�N                                    Bxg���  "          @�@\)��ff>Ǯ@Q�C���@\)��{�#33��33C�XR                                    Bxg�n  
�          @�(��AG���{@��RB#Q�Cj33�AG����@ffA�CrE                                    Bxg�  
�          @�G�?�p���\)?�\)Ag�C��?�p���\���H�3\)C��3                                    Bxg� �  �          @�R?�p����@*�HA�(�C�Ff?�p����Ϳ����33C��)                                    Bxg�/`  "          @�?����\)?#�
@�{C��?�������#33��Q�C���                                    Bxg�>  
�          @���>�����p���(��P��C�C�>������
����=qC���                                    Bxg�L�  
�          @�?+����H>�
=@C33C�q�?+���ff�G�����C���                                    Bxg�[R  
�          @�
=?�G���?h��@��
C���?�G��߮�{��
=C��                                    Bxg�i�  
�          @�p�@
=��G�?}p�@��C�@
=��
=�
=���C�h�                                    Bxg�x�            @�{?�Q���?Tz�@ǮC�]q?�Q���
=�!���Q�C���                                    BxgÇD  
n          @��?�z���G�?��RA�C�1�?�z�����\)��33C�b�                                    BxgÕ�  "          @��?�����(�?E�@���C��?����޸R�%��=qC�U�                                    Bxgä�  "          @�z�>��H���?(��@��RC���>��H��=q�0����Q�C��)                                    Bxgó6  �          @�@O\)����@'
=A�(�C��@O\)��G���G��VffC�                                    Bxg���  �          @�33@z�����?��Ad(�C���@z���Q쿪=q�)G�C�w
                                    Bxg�Ђ  
�          @����ff�񙚿.{��z�C�f��ff��
=�|����ffC�^�                                    Bxg��(  �          @�׿˅��G��u�   C��ÿ˅��Q��QG���{C�!H                                    Bxg���  �          @�Q�?��
��33@W
=A�z�C�s3?��
��\)>�G�@h��C�c�                                    Bxg��t  
�          @��H?������@0  A��C�E?����\)�u� ��C���                                    Bxg�  h          @ۅ��R���?�ffAz=qC�=q��R�љ������$��C�L�                                    Bxg��  
Z          @�׿��׮?���A9G�C��\��������H�eC���                                    Bxg�(f  
�          @��
�L���ƸR?�AV�HCr� �L����녿�(����Cs!H                                    Bxg�7  
�          @�ff�'
=����?�\)AC�Cy��'
=��G���\)�B�\Cy�                                    Bxg�E�  S          @�\����z�@�HA�
=C}�
����  �Tz����C~aH                                    Bxg�TX  �          @�Q�����G�@E�A�(�C}W
����ff�B�\���C~�                                     Bxg�b�  	�          @����   �˅@|(�A��HC|�{�   ��{?G�@���C~�q                                    Bxg�q�  A          @�z��{�љ�@|(�A�ffC~!H��{���
?333@��\C�                                      BxgĀJ  g          @���\)�θR@�(�B�C���\)���\?�{A��C��3                                    BxgĎ�  
�          @�����p�@�G�B��C�j=�����?\A9C�s3                                    Bxgĝ�  �          @��R�����=q@j�HA�\)Cx�\�������?�@�
=C{h�                                    BxgĬ<  "          @�Q�������@W
=A�33Cy�f������
>8Q�?�\)C{�                                    Bxgĺ�  
�          @��H�i����  ?�{A(�Clp��i����{�����>�\Cl#�                                    Bxg�Ɉ  
�          @���?^�R�9��@�33BV�C�aH?^�R��z�@��A�\)C��H                                    Bxg��.  "          @���?Q��,(�@a�BM�C�w
?Q��|��@�AՅC���                                    Bxg���  
�          @���fff�>�R@s�
BK
=C}.�fff���\@
=qA�  C�                                      Bxg��z  T          @����ÿ���O\)��C\n���ÿ\(�������CP�                                    Bxg�   "          @�G���\)�������;��CAǮ��\)?������;z�C%�H                                    Bxg��  
�          @�
=��(�������@Q�C9���(�?�������0�HC:�                                    Bxg�!l  
Z          @����������,(�C9���?�Q���z�� p�C"(�                                    Bxg�0  
�          @�Q���G��#�
�����.��C4
��G�@��������C��                                    Bxg�>�  T          @�������?�ff����:z�C(�����@Tz����\�33C33                                    Bxg�M^  T          @��H���?�=q����1�C(�����@S�
��Q��z�C�H                                    Bxg�\  	�          @����ff?=p����\�!{C,�f��ff@5������
=C�=                                    Bxg�j�  T          @�z���?O\)��=q�C,� ��@1�������
=C                                    Bxg�yP  
(          @��
���
?��R������RC(����
@I���s33��33C�q                                    BxgŇ�  �          @���{>L����\)�/\)C1����{@�\���
=C+�                                    BxgŖ�  
�          @�����>���{����C0���@�\�X�����C!�                                    BxgťB  
�          @�����
�Ǯ��z��C8  ���
?�  �������C%\                                    Bxgų�  
�          @�  ���������z��G�C?Q�����?c�
����HC*xR                                    Bxg�  
(          @޸R��33�5�fff���C:�f��33?aG��dz����RC+}q                                    Bxg��4  �          @�33��{���
��ff��CA5���{?(���33���C-�H                                    Bxg���  T          @�����z�@8�ÿ�=q�q�C�\��z�@X�þ�G��hQ�C)                                    Bxg��  "          @�����
@���p�����Cu����
@L�Ϳ�p����CY�                                    Bxg��&  "          @�\��(�?�ff�1G���{C(�H��(�@��(��y�C ��                                    Bxg��  A          @���33>L���$z����\C2^���33?�  �  ���C)��                                    Bxg�r  
;          @�Q����ÿ�\)�G���(�C=ff���ý�\)�!���C4�)                                    Bxg�)  "          @����Ϯ��(���{�t��CD�f�Ϯ���\� ����G�C<��                                    Bxg�7�  	�          @�z����>�{�N{��z�C0xR���?�33�2�\��\)C#n                                    Bxg�Fd  �          @ٙ���\)�L���Y����\)C;����\)?.{�[���\)C-33                                    Bxg�U
  "          @���G��^{��
=��  CUJ=��G��
=�L����p�CL                                    Bxg�c�  T          @Ӆ�����  ���R�7
=C\������HQ��2�\�ӅCU�)                                    Bxg�rV  
�          @�(�@z���p��
=q��=qC��@z���\)�(Q����\C�c�                                    Bxgƀ�  
�          @�����  ��\)�z����
C~�ÿ�  ��  �333��z�C|�                                     BxgƏ�  �          @Ǯ�33��ff���H��\)Cx���33��G��%���
Cu��                                    BxgƞH  
�          @�{�������xQ��33C��������{�R�\�33C�˅                                    BxgƬ�  "          @����.{��p��(����CrE�.{��ff�.�R��
=Cn�=                                    Bxgƻ�  	�          @Ǯ������(��k���Cd��������(��
=����Ca��                                    Bxg��:  T          @�����\)�L�;8Q��=qCR@ ��\)�5��(��V=qCOk�                                    Bxg���  
(          @Ӆ��p��?\)?�=qA;33CO����p��QG��#�
����CQ��                                    Bxg��  
�          @љ���33�{?�ffA�=qCH�q��33�0��?=p�@ϮCM:�                                    Bxg��,  	�          @�  ��{��\@Tz�B{CH����{�>{@�
A���CT\)                                    Bxg��  "          @��Ϳ�z��O\)@ ��B��Cvuÿ�z��~�R?xQ�AP  Czz�                                    Bxg�x  T          @�G��ff��z�?��A((�Cvs3�ff�����Q��8z�Cv\)                                    Bxg�"  
�          @θR�������@�z�Ba�Ch�3�������
@O\)B��Cu�{                                    Bxg�0�  "          @���������\?�33A��Cp!H�����*=q>�33@�
=Cs��                                    Bxg�?j  T          @��R�QG����׿Y���  Ch�{�QG��c�
� ����\)Ccu�                                    Bxg�N  "          @���<����(��xQ����Cn�)�<�����\�7
=��\Cj)                                    Bxg�\�  
(          @��
�Ǯ��\)?�A�ffC}p��Ǯ���R�z���=qC~�                                    Bxg�k\  T          @��ÿ���U@�ffBb�Cxٚ�����  @h��BC��                                    Bxg�z  
�          @�ff�333�.�R?J=qA7
=C`G��333�4z᾽p����
Ca0�                                    Bxgǈ�  "          @�������)���0  ���
CQ�R���Ϳ����e�\)CC��                                    BxgǗN  �          @�����H�s�
��  �C�C]J=���H�?\)�*�H��33CV��                                    Bxgǥ�  
�          @�Q��Vff�C33>�z�@hQ�C^E�Vff�:=q�u�BffC\�                                    BxgǴ�  "          @����E�{��q��p�Cg�)�E���G��U�CV&f                                    Bxg��@  �          @ʏ\��G����H�J=q�G�C�
��G���(��*=q� 
=C��                                    Bxg���  
�          @Å��
=�y������v{Cy�)��
=�G��&ff��\Cuc�                                    Bxg���  �          @�G����R��@~{BC�E���R�ʏ\?�=qA\��C�u�                                    Bxg��2  �          @�����  @(�A�=qC�AH�����>��@*=qC�T{                                    Bxg���  
�          @�G���\)�Z�H�x���3�Cq=q��\)�˅���}�HC\aH                                    Bxg�~  �          @�Q쿺�H���\�~�R��C|T{���H�(Q���
=�t��Cp��                                    Bxg�$  �          @�(���=q�\)��=q�YG�C�
=��=q��(���ff�)C���                                    Bxg�)�  
Z          @�{@(���p���=q�7(�C�/\@(���
������C�Z�                                    Bxg�8p  
�          @�(�@����ff��
=�5G�C�9�@�������z�ǮC���                                    Bxg�G  �          @�?\��p���z��p�C�|)?\�X���Ϯ�q  C��                                    Bxg�U�  
(          @�{?   ��Q��Z�H��RC�H�?   ��p���33�NG�C�:�                                    Bxg�db  T          @�  ?�ff������z��.G�C���?�ff�ff��G��p�C��q                                    Bxg�s  T          @�{��z�h������CT�)��z�?h����\)�C\                                    Bxgȁ�  
�          @��H�~�R@QG����H���C�H�~�R@���\)��{C�)                                    BxgȐT  
�          @���@��k��	G�C(���@\(��"�\��\)C�                                    BxgȞ�  �          @�G���
=>L���.�R��\)C2.��
=?��R�(���{C(�3                                    Bxgȭ�  "          @�G���{?G��(Q���z�C-���{?޸R�����C$�                                    BxgȼF  "          @�p���z�?fff���iG�C,�\��z�?��
�����)G�C'}q                                    Bxg���  
�          @����?5��\��Q�C-�����?��������^�\C'�                                    Bxg�ْ  T          @ٙ���\)?��H�����U��C)u���\)?�(��z�H���C%(�                                    Bxg��8  T          @����G�?��ÿ�(���Q�C*�R��G�?޸R��33�;�C%\                                    Bxg���  "          @߮��{?��=q��{C)����{@ �׿��
�l(�C"��                                    Bxg��  �          @��H����?����z���  C(������@ �׿�z��7\)C#+�                                    Bxg�*  �          @��
��\?�����=q�&�RC(k���\?��Ϳ+����C%\)                                    Bxg�"�  
�          @�{��ff?�
=�z����C&s3��ff?�\=���?Q�C%��                                    Bxg�1v  
�          @޸R��(�@��=���?Y��C")��(�?�Q�?h��@���C#�3                                    Bxg�@  "          @߮��p�@.{>\)?�
=C{��p�@p�?�AffC�                                    Bxg�N�  T          @����=q@XQ�?xQ�@�C޸��=q@0��@�A�
=C��                                    Bxg�]h  �          @޸R���@Mp�@�A��C5����@z�@Z=qA�C�q                                    Bxg�l  
�          @Ϯ���\@e��.{���C����\@dz�?:�H@θRC.                                    Bxg�z�  
�          @Ӆ���@R�\@A��\C�����@  @K�A�G�C)                                    BxgɉZ  T          @�
=?��Ϳ�z�@���B��qC��)?����hQ�@��BTffC�33                                    Bxgɘ   �          @��H@e���\@�BS=qC�&f@e��|(�@�Q�B�C�!H                                    Bxgɦ�  "          @�33@aG��Ǯ@�p�B_(�C�f@aG��e�@��B+p�C�@                                     BxgɵL  
�          @�z�@S�
��  @��
Bk��C�� @S�
�W�@�
=B9�
C�@                                     Bxg���  
�          @��
@S33�=p�@�ffBqz�C��R@S33�:=q@��BG��C�J=                                    Bxg�Ҙ  
�          @�ff?O\)>��@��HB�{A�
=?O\)��\@�  B�8RC�ٚ                                    Bxg��>  
�          @�{����?J=q@��HB��B�=q���Ϳ�ff@�ffB���C�f                                    Bxg���  �          @�ff��33����@��B�p�C`xR��33�P��@�p�BL�Cv��                                    Bxg���  
�          @��þ��?s33@��
B�#�B��H����˅@�  B�.C�T{                                    Bxg�0  T          @�ff?
=?���@׮B��B��R?
=��ff@ڏ\B�C���                                    Bxg��  
�          @ۅ?�?���@�
=B�Bdz�?��5@�B�C�]q                                    Bxg�*|  
Z          @�33>�Q�?Ǯ@�\)B�ffB��
>�Q�z�@�p�B�B�C��                                    Bxg�9"  �          @�33��
=��
=�G���p�C�H��
=��Q��;���z�C��)                                    Bxg�G�  �          @�{��
=��=q�����_33C��쿗
=�����tz���C���                                    Bxg�Vn  "          @��Ϳ�\)���Ϳ�\)�Z{C}����\)��33�w
=��Cz�3                                    Bxg�e  T          @ᙚ�\)�������-p�Cx޸�\)��  �e��(�Cu��                                    Bxg�s�  �          @�(��"�\�\?��A~ffCwL��"�\��33������Cx(�                                    Bxgʂ`  
�          @��H�A����@=qA�\)Cq�
�A���33>8Q�?�ffCs�\                                    Bxgʑ  "          @�=q��R����@g
=A��RCx{��R�Ϯ?�=qA/�C{                                      Bxgʟ�  �          @���`  ����@��B(�Cf�R�`  ��=q@   A�z�Cm��                                    BxgʮR  "          @�\)������@�33B/�\Ctff����Q�@7
=A��Cz�                                    Bxgʼ�  �          @��
���
��(�@�{B9(�Cw�)���
���
@@��A�33C}!H                                    Bxg�˞  T          @��
��
�}p�@��B?G�Co����
����@Q�A�CwG�                                    Bxg��D  
�          @����C�
@�  Bs�CqW
�����@�
=B%(�C{�f                                    Bxg���  �          @�ff?0�׿+�@ҏ\B�B�C��?0���@��@�z�Byp�C�~�                                    Bxg���  �          @��?�{>�{@���B�
=Ab�R?�{�
�H@��B�aHC��                                    Bxg�6  "          @�z�?�
=?(�@�  B�� AظR?�
=����@�  B���C��R                                    Bxg��  T          @�G�?��>���@��B��A'�
?���z�@�\)B�
=C�%                                    Bxg�#�  "          @ٙ�@G�>�\)@ʏ\B�Q�@��
@G�� ��@�Q�B�L�C��\                                    Bxg�2(  "          @�ff?˅�C�
@�Bk  C��?˅��z�@��RB=qC��R                                    Bxg�@�  �          @�  @Q��@��@�=qBb{C��=@Q�����@�(�B��C���                                    Bxg�Ot  T          @��@J=q���@�
=BdC��=@J=q�s�
@�p�B-G�C���                                    Bxg�^  
Z          @�\)@I���:�H@�z�Bt�HC�t{@I���1�@��BM(�C�J=                                    Bxg�l�  	�          @�Q�?�ff�!G�@��B{��C���?�ff���\@���B.ffC�Y�                                    Bxg�{f  
�          @�
=?��R�c33@�p�BZp�C�^�?��R���@s�
B�
C��                                    Bxgˊ  "          @��?^�R�i��@�BXffC��
?^�R���@c33B�C��
                                    Bxg˘�  T          @�Q�?���J=q@��Bf�C�:�?�����
@|(�B(�C���                                    Bxg˧X  
�          @�  ?�ff��ff@�p�B��HC��?�ff�S33@��\BV  C�O\                                    Bxg˵�  T          @ҏ\?
=q?.{@�p�B��HBN
=?
=q���
@���B���C�˅                                    Bxg�Ĥ  T          @�Q��ͽL��@�{B��fC9�3����33@��B�u�C~��                                    Bxg��J  �          @Å?��@Q�@�z�B|�BtQ�?��>�{@�(�B�  Ah��                                    Bxg���  
�          @Å?��H?�@�33B�k�A�  ?��H�Ǯ@��B���C��                                     Bxg��  �          @�G�>���R@�B�z�C�T{>��#�
@��B{�C�)                                    Bxg��<  �          @Å��R�p�@�=qBb�Cc�
��R��=q@x��B�
CqJ=                                    Bxg��  �          @��ͼ��
��\@��RB�L�C�"����
�#33@�p�B���C��{                                    Bxg��  �          @�ff������
=@�33B�
=C`�������R�\@�Q�BS�
Cv5�                                    Bxg�+.  
�          @�G������33@n�RB��Czs3������
?��RA��HC}�3                                    Bxg�9�  �          @�(����R�r�\@��BE\)Cx�=���R��{@H��A�G�C~                                      Bxg�Hz  �          @�녿}p����
@h��Bz�C��}p����H?�=qA���C��R                                    Bxg�W   T          @�=q���
��G�?�Q�A�=qC}�{���
�����u�ffC~�=                                    Bxg�e�  �          @�(�?aG���@�33B)��C�S3?aG���(�@�RA���C�@                                     Bxg�tl  T          @�ff?�R���R@��
B0�C��f?�R��Q�@.�RA£�C��
                                    Bxg̃  "          @�
=��p��<(�@���Br�
C�e��p����@}p�B#G�C���                                    Bxg̑�  T          @��H�L����@�B  C{�ͿL������@�G�B1��C�l�                                    Bxg̠^  "          @�\)�N�R�+�@��
B2
=C[��N�R�z�H@=p�A�(�Cf�=                                    Bxg̯  
�          @��H��
��@�z�Bc�C_^���
�mp�@vffB%��Cn�                                    Bxg̽�  "          @�
=������
@���Bs=qCp33�����r�\@l��B*�Cz�                                     Bxg��P  �          @�Q�� ���QG�@P  B!(�Cn� � ����?���A��HCt^�                                    Bxg���  �          @��
�h���Tz�?�ffA��C^W
�h���i��>�{@qG�Ca�                                    Bxg��  T          @�\)�e��\��=q�L\)CJ���e�>B�\�����Zp�C0��                                    Bxg��B  "          @������
��33����-��CF�R���
=�\)���H�:{C3
=                                    Bxg��  �          @������H@{�333��ffC�=���H@A녿����G�Cz�                                    Bxg��  "          @��\�%�@�=q�W
=�(�B�Q��%�@�=q?Q�Az�B�G�                                    Bxg�$4  �          @�  �aG�@7
=�����'�C��aG�@����7
=��RC                                    Bxg�2�  �          @���J=q@&ff�����<Q�Cu��J=q@z�H�Q��G�C �\                                    Bxg�A�  �          @˅����@*=q@\)A�
=C������?�@O\)B �C!�                                    Bxg�P&  T          @������?�(���(��/�
C(�q����?Ǯ�=p����C%�                                    Bxg�^�  �          @�
=��Q�?�������C$&f��Q�@  ��Q��|��CǮ                                    Bxg�mr  !          @���K�@�\?+�A"{CJ=�K�?ٙ�?��A�  C��                                    Bxg�|  
�          @�\)��33@3�
?��HA�(�C����33@@#33A癚C                                    Bxg͊�  
�          @��H�U�@��\@(�A�Q�B�.�U�@aG�@s�
B��Cz�                                    Bxg͙d  "          @�p����H@��
?�
=A#\)C �����H@�{@333AƸRC��                                    Bxgͨ
  �          @����z�@���?�@�\)C� ��z�@w�?�z�A��\C
��                                    BxgͶ�  �          @ָR���
@b�\�(���\)CY����
@�zΐ33�=qC                                    Bxg��V  �          @ڏ\��Q�@k������\C���Q�@�
=�u� ��CG�                                    Bxg���  
�          @�  ����@_\)�+����C\)����@�p���33�AG�Ck�                                    Bxg��  
�          @���{@!G��N{��\)C���{@[��{��p�C�=                                    Bxg��H  "          @�������@9����(����RCJ=����@S33�.{��\)C��                                    Bxg���  
�          @��R�W�@��?\(�A\)B��
�W�@�ff@ffA��HC �R                                    Bxg��  "          @��
��ff@~{>.{?�z�C�
��ff@mp�?�Q�Ag33C
�{                                    Bxg�:  
�          @ҏ\����@QG������RC}q����@[�=�\)?�RCE                                    Bxg�+�  
�          @����33@W
=�������C���33@a�=�\)?�RC��                                    Bxg�:�  	�          @�Q���
=���
�aG���C7)��
=?Q��\(���p�C,&f                                    Bxg�I,  T          @�����
=�,(����\�{CS{��
=���R��\)�633CC�=                                    Bxg�W�  
Z          @ٙ����+������=qCS#�����  ����5p�CC�                                    Bxg�fx  
�          @ڏ\���ÿ����e� �CC�{���þ�{�z=q���C7�                                     Bxg�u  
�          @ٙ������(������HCO����ÿ�  ���/
=C?�)                                    Bxg΃�  	�          @ָR��{��G��k����C>xR��{>aG��s�
�  C1�3                                    BxgΒj  "          @�z������@  �����%\)C<�
����?&ff��G��&�C,O\                                    BxgΡ  
(          @�{�^�R�h�������d(�CB�R�^�R?c�
�����dp�C%�f                                    Bxgί�  "          @�G��|�Ϳ�����
=�M�CGJ=�|��>�{��z��V��C/)                                    Bxgξ\  t          @�
=����fff�\)���C=� ���>�����=q�{C/�
                                    Bxg��  
Z          @׮�����������!�CF8R��������z��.��C5�q                                    Bxg�ۨ            @�z���{�&ff�i����
C:Ǯ��{?��j�H���C.ff                                    Bxg��N  
(          @�
=��{�\(�����3�
C;����{��녿�  �P��C7��                                    Bxg���  T          @�Q���녿�Q��{��=qC@�����=�Q���33�C2��                                    Bxg��  �          @�G���
=�aG������J�\C7���
=?����
=�@�C!�R                                    Bxg�@  �          @�Q��Tz�>�����33�q(�C.T{�Tz�@Q����R�Xp�CE                                    Bxg�$�  �          @�
=����Q������*Q�C>#����?����33�,p�C-(�                                    Bxg�3�  T          @�{��33��Q��;���=qCG&f��33�p���XQ���ffC=�=                                    Bxg�B2  �          @�G���G��_\)�Dz���CY���G�����  ��HCO�                                    Bxg�P�  �          @�G���z��Y���j=q�	G�C[s3��z��
=�����1(�CO                                      Bxg�_~  
�          @������\�I���l(��
33CX\���\��{��\)�.(�CKG�                                    Bxg�n$  
�          @љ������G������CY�{���׿�(���=q�?
=CK5�                                    Bxg�|�  �          @θR����L(��p  �\)CY�������������4�HCL��                                    Bxgϋp  �          @�����������p���=qCB�3����\)�1G����C:+�                                    BxgϚ  T          @Ϯ��Q�\�`  �33C8!H��Q�?5�\���  C,Y�                                    BxgϨ�  T          @�ff�z=q��p���\)�O
=C9p��z=q?�z�����G��C#k�                                    BxgϷb  �          @�=q��
=?��@  ��C'�3��
=@��   ���C��                                    Bxg��  �          @ƸR��\)@C�
��p��k�
C!H��\)@W
=������C��                                    Bxg�Ԯ  �          @�z���33�E��c�
�"  C>����33>�{�hQ��%�RC/@                                     Bxg��T  
�          @�=q��  ��(���z��)G�CH���  ���
��ff�8C8�                                    Bxg���  "          @�����p���
=�u���\CF����p���ff���� p�C98R                                    Bxg� �  "          @�{��  >u�����4=qC0����  ?˅����&G�C �=                                    Bxg�F  
Z          @Ϯ�qG�?Tz���33�M�HC'���qG�@  ���
�3�\C#�                                    Bxg��  �          @У����H����Z=q��HCD�)���H��ff�n{�G�C9�                                    Bxg�,�  
(          @�p���  �5���R����CQ� ��  ���H�Mp����CIaH                                    Bxg�;8  "          @�p���녿&ff�k����C;�f���>��l�����C.L�                                    Bxg�I�  �          @�=q�|��@Z=q�s�
��C8R�|��@�z��(Q����C�q                                    Bxg�X�  
Z          @Ϯ��Q�?����k��33C�3��Q�@7
=�=p�����C�                                    Bxg�g*  "          @���  @L���G����C���  @~�R�����C��                                    Bxg�u�  
�          @�G���R@�=q���R���B�����R@�p����
�?\)B�8R                                    BxgЄv  
�          @�
=��  @j�H��  ����C+���  @��ÿ
=��Q�C��                                    BxgГ  �          @�
=����@9����33�|z�C�����@P�׿8Q���=qC.                                    BxgС�  �          @���S33@(���.�R�\)CT{�S33@Tz��=q����C�                                    Bxgаh  T          @����J=q@�z�������B��\�J=q@�G��&ff��Q�B��)                                    Bxgп  "          @�{����@1G���z��i(�B�aH����@�\)��33�-{B�33                                    Bxg�ʹ  "          @�33����@�Q���  �B�aH����@�\)� �����
B���                                    Bxg��Z  
�          @���@��R�
=q��{B�����@�33�����33B�=q                                    Bxg��   
Z          @У��=q@~�R��p��.{B�z��=q@��H�Dz���RB�R                                    Bxg���  
�          @�녿���@����
=�!��B�  ����@�(��,����  B׸R                                    Bxg�L  �          @����tz�@g
=�|����HC���tz�@�33�/\)��
=C��                                    Bxg��  
�          @Ӆ�p  @qG��l���Q�C޸�p  @�p������  C �q                                    Bxg�%�  �          @����p�@Dz��1G��ŮCB���p�@o\)��G��w33C)                                    Bxg�4>  �          @ָR���@�  �E��z�C����@��R��p��p��C��                                    Bxg�B�  "          @�ff�u@����O\)��(�C)�u@�z����  B�u�                                    Bxg�Q�  �          @�
=�x��@������  C ���x��@�=q�W
=��  B�aH                                    Bxg�`0  
�          @�G���=q?������H�+�HC���=q@E�w��G�C��                                    Bxg�n�  �          @�\)��\)@P  �8����
=C�f��\)@|(�����G�C��                                    Bxg�}|  "          @�  ��Q�?Ǯ�C�
��p�C$�H��Q�@
=�\)����C�                                    Bxgь"  �          @�����@X�ÿ�(��V�\C�
����@j�H��ff����C}q                                    Bxgњ�  
�          @����G�@p��&ff��(�C5���G�@7
=��{��
=C�3                                    Bxgѩn  
�          @�z���{?:�H�S33��33C,����{?У��=p���ffC#��                                    BxgѸ  �          @�=q���
>8Q��8����p�C2B����
?}p��.{��ffC*xR                                    Bxg�ƺ  �          @��
��(�=L�Ϳ�(��O�C3����(�>���33�D��C/�
                                    Bxg��`  
Z          @�33��z�?\(�����33C,  ��z�?�Q��������C&�=                                    Bxg��  �          @�33��(�?��z���\)C.����(�?����z�����C(�                                    Bxg��  �          @�(����H?�������=��C*5����H?�
=�}p��
�\C'O\                                    Bxg�R  T          @Ӆ��Q�?�ff��G��0��C&
��Q�?��G��ٙ�C#�)                                    Bxg��  T          @��
���?\����a�C&�����?���<�>�z�C&:�                                    Bxg��  �          @�33�ҏ\>�p�����z�C0�q�ҏ\>��ͼ��
�W
=C0�=                                    Bxg�-D  "          @�p�����>�p�    <�C0�����>�Q�=�G�?k�C0��                                    Bxg�;�  "          @�33��Q�=�G��L����G�C3���Q�>��ÿ=p���  C1!H                                    Bxg�J�  "          @Ӆ��G�?:�H�Ǯ�X��C-����G�?O\)�.{��G�C,�3                                    Bxg�Y6  
�          @�33��(�?�p��W
=��ffC&���(�?�p�>W
=?�C&�                                    Bxg�g�  
�          @љ��ƸR?�ff�+���ffC#���ƸR?�������C"�
                                    Bxg�v�  �          @Ӆ�˅?�p�>��@��C&��˅?���?&ff@��C(�                                    Bxg҅(  �          @�=q���R?��?�z�As
=C'�{���R?O\)?���A��C,E                                    Bxgғ�  T          @�33�θR?�R?J=q@�p�C.}q�θR>\?n{A�\C0�
                                    BxgҢt  
�          @���Ϯ?\(��\�W
=C,k��Ϯ?p�׾���C+��                                    Bxgұ  �          @�(����>k��B�\�ӅC1�R���>�(��+����C0=q                                    Bxgҿ�  
�          @�=q�Ϯ�   ����(�C8^��Ϯ���ÿ#�
����C6�                                    Bxg��f  �          @�Q���ff��(���G��w
=C7�\��ff�����
=q��  C6��                                    Bxg��  �          @�  �ə��\)>��
@9��C9
=�ə���R>#�
?�
=C9��                                    Bxg��  �          @�=q��
=�������C5)��
=>�ff��\����C/�f                                    Bxg��X  T          @Ӆ�ƸR>\�	����Q�C0z��ƸR?s33��(�����C+W
                                    Bxg��  �          @ҏ\����?��
�ff��=qC*Q�����?�\)���R��G�C$�                                    Bxg��  T          @�G����?����E��=qC ����@'
=��R���
C�=                                    Bxg�&J  �          @����?�(��-p���33C&{��@���p����HC &f                                    Bxg�4�  �          @Ӆ��Q�?�G��)�����
C#
=��Q�@�������CaH                                    Bxg�C�  "          @������@0���S33��ffC!H���@b�\�����z�C��                                    Bxg�R<  	�          @Ӆ���R@��3�
��=qB͸R���R@�\)��G��;33B�k�                                    Bxg�`�  
�          @�z��z�@���\)��33B����z�@�(��#�
��G�B�k�                                    Bxg�o�  �          @�(���Q�@�{�J�H�噚C���Q�@�(�������\)Cn                                    Bxg�~.  T          @ҏ\���\@u��'
=����C
=���\@�z��  �U��C33                                    Bxgӌ�  �          @�������@p  �e��p�C������@���{����C^�                                    Bxgӛz  T          @�{��\)@N{�`����
C:���\)@�G��"�\���\C	��                                    BxgӪ   "          @�(���=q@.{������
C����=q@N{���H�Lz�C�q                                    BxgӸ�  �          @�33���?��H�1���
=C(J=���?�33�
=���C"
=                                    Bxg��l  "          @�{��(�?z��%����C.����(�?��R���  C(�\                                    Bxg��  �          @���{�\)�(���{C5E��{?   �����z�C/W
                                    Bxg��  �          @�33����@%�i����CQ�����@\���6ff��C+�                                    Bxg��^  "          @������?��H�e���=qC ������@4z��<(���Q�C�                                    Bxg�  �          @�Q���Q�?�
=�dz���Q�C&��Q�@�
�E�����C�                                    Bxg��  "          @���ָR���W����C5��ָR?@  �Q���p�C-�
                                    Bxg�P  �          @���Ӆ��33�E�����C=�{�Ӆ���R�QG���(�C6�3                                    Bxg�-�  �          @����  �B�\�%���Q�C:c���  �u�,(�����C4�=                                    Bxg�<�  �          @�33����?���a���(�C/�����?�Q��QG���C'Y�                                    Bxg�KB  �          @��H����.{�u���
=C5�)���?Q��p  ����C,p�                                    Bxg�Y�  
�          @�Q���p��p���J=q�ϙ�C<Q���p������R�\��33C4޸                                    Bxg�h�  "          @����ff�����5��  C4�f��ff?!G��1G����C.n                                    Bxg�w4  �          @�33���k��<������C6��?��:=q��(�C/@                                     Bxgԅ�  "          @���Ǯ?�z��/\)��(�C%)�Ǯ@33��R��=qC�q                                    BxgԔ�  "          @�ff��=q�L���C�
��=qC;xR��=q<#�
�J=q���C3�                                    Bxgԣ&  "          @����>��h����ffC2�3���?�ff�^�R��\C)�{                                    BxgԱ�  "          @�p�����<��Z�H��=qC3�q����?c�
�S33�ۙ�C+��                                    Bxg��r  T          @�{��\)>8Q��u���C2E��\)?�33�j�H����C)!H                                    Bxg��  "          @����>\)��p��0��C2����?�Q���
=�(33C$ٚ                                    Bxg�ݾ  �          @����33�#�
�����2z�C4n��33?�����
�+�RC&k�                                    Bxg��d  �          @�z���G������Tz��ޣ�C6����G�?���R�\��=qC/                                      Bxg��
  �          @�z����ÿz��1G���33C9\����>\)�5���33C2��                                    Bxg�	�  �          @����
���
�/\)��(�C4+����
?+��*=q��z�C.:�                                    Bxg�V  T          @�����
=�
=�<(��£�C90���
=>.{�?\)��z�C2��                                    Bxg�&�  �          @����(������G��	ffC9�\��(�>�(�����
33C/��                                    Bxg�5�  "          @ᙚ��
=�����Z=q��z�CA����
=����j=q����C9E                                    Bxg�DH  "          @޸R���R�8Q��\������C:�f���R>#�
�aG���=qC2z�                                    Bxg�R�  �          @�������Ǯ�\����C7� ���>��\(�����C/Y�                                    Bxg�a�  �          @�p����
��Q��A��ҸRC4�\���
?+��<����G�C-Ǯ                                    Bxg�p:  "          @��H��\)���H�����>=qCA0���\)��=q���l(�C=ٚ                                    Bxg�~�  �          @�G���p����ÿfff���RCDxR��p���ff�����;�
CB\                                    BxgՍ�  �          @��H��(��	���z�H���CGQ���(�����p��O�CD��                                    Bxg՜,  �          @���\��
�����?�CHǮ�\��33��33��33CET{                                    Bxgժ�  �          @�  ���H��\�ٙ��j�RCH�f���H���p���CD�                                    Bxgչx  �          @׮����9������  CN0�����#�
��(��o�CKp�                                    Bxg��  "          @�  ��(��8�ÿ��DQ�CN5���(��p����G�CJ��                                    Bxg���  �          @�\)�����G���
=�G
=CF0����Ϳ�{������CB�                                    Bxg��j  "          @����ҏ\�!G����H�F�HC9s3�ҏ\��  �����V{C65�                                    Bxg��  "          @�  ��
=�������XQ�C6+���
=�#�
���s�
C5G�                                    Bxg��  �          @�{��z�>\��R���
C0�)��z�?���\���RC/��                                    Bxg�\  T          @�{�ڏ\?��;����QG�C*�
�ڏ\?�z���xQ�C*T{                                    Bxg�   
�          @ڏ\��Q�?333��\�qC-�{��Q�?�{����R�RC*Q�                                    Bxg�.�  T          @��
���
?�(�����P(�C"�=���
@�\���\���C B�                                    Bxg�=N  �          @�Q�����@:�H�G���(�C33����@AG��L�;�Ck�                                    Bxg�K�  �          @ٙ��ʏ\?�R��
=��p�C.h��ʏ\?��ÿ�p��qG�C*n                                    Bxg�Z�  �          @��
��p�?\)�����C/��p�?���z���p�C*^�                                    Bxg�i@  �          @أ���{?\(��
=��\)C,���{?��Ϳ���C'�q                                    Bxg�w�  �          @�  �ҏ\?�G����s�C'��ҏ\?�z῵�;�
C#��                                    Bxgֆ�  �          @ָR���?�������\C#^����@\)����?\)C �                                    Bxg֕2  !          @�
=��G�?��
����{C"����G�@\)�У��k�C��                                    Bxg֣�  �          @У����@5����(��C@ ���@AG�������C��                                    Bxgֲ~  �          @�{��Q�@A�?˅A{�Cz���Q�@$z�@��A�C�H                                    Bxg��$  �          @�����@%���
=�j=qC�R���@&ff>W
=?�C�                                    Bxg���  �          @�z���\)@\��?��
A>�HCB���\)@C33@�
A��C�                                     Bxg��p  �          @�����p�@8��@(�A�p�C�
��p�@�R@C33A�  C�H                                    Bxg��  �          @�G���Q�?�p�@|(�B
=C#����Q�?�\@�B�C.0�                                    Bxg���  "          @�������?�@w�B(�C.^��������@x��BC8^�                                    Bxg�
b  �          @�=q��z�@
�H?k�Ap�C� ��z�?��?�33AL��C"(�                                    Bxg�  �          @�z����\@����+����C�����\@�������P��C�)                                    Bxg�'�  �          @޸R��
=@����Q���C
��
=@��þB�\����C                                      Bxg�6T  �          @�p�����@c�
��p��$��CQ�����@p  ��p��C33C��                                    Bxg�D�  �          @޸R��=q@333��G��lz�CB���=q@HQ쿇��G�C��                                    Bxg�S�  �          @�p�����@&ff�7���ffC������@L��������C)                                    Bxg�bF  T          @�\��
=@{�Z�H���HC��
=@=p��333��Q�C��                                    Bxg�p�  �          @޸R����?
=q�I����Q�C.�)����?�G��<(���Q�C(0�                                    Bxg��  �          @����s33@�\)�mp��33CW
�s33@�
=�$z����B�.                                    Bxg׎8  �          @�����@A���
=�#��C  ��@\)�i����CQ�                                    Bxgל�  �          @��
�+�@����J=q��ffB�q�+�@�(�������
B�                                    Bxg׫�  T          @���W
=@��
�8����ffB��{�W
=@��ͿУ��_\)B�                                     Bxg׺*  �          @�33��\)@QG������\)CJ=��\)@��
�K���z�C��                                    Bxg���  �          @�z���?�p��,����z�C%����@�
�����C �{                                    Bxg��v  �          @�  ��\)?�z��%����\C%���\)@p��Q����HC s3                                    Bxg��  �          @�����
=?��n�R���RC/B���
=?�{�`�����HC'��                                    Bxg���  �          @陚����>��
�e��C1{����?����Z�H��G�C)�                                    Bxg�h  �          @���\)=�Q��{���
C3=q��\)?
=�����z�C.��                                    Bxg�  �          @�=q��33?��
?h��@�p�C)��33?��?�A�C+�                                    Bxg� �  �          @�������?Y���
=���C,O\����?�{�ff����C'�                                     Bxg�/Z  �          @�����?:�H�������C,(�����?�33�xQ��33C"��                                    Bxg�>   
�          @�Q��ҏ\�
==��
?5C9)�ҏ\�
=�L�;��C9#�                                    Bxg�L�  T          @ٙ�����Vff@mp�B�HCXO\������@6ffA�33C]�q                                    Bxg�[L  �          @�G������	��@<(�A��CIG������0��@�A�ffCN��                                    Bxg�i�  "          @�Q������@A��HCE������?���A�\)CI��                                    Bxg�x�  T          @ڏ\�����.{@1�A£�CNE�����Q�@A��\CR�q                                    Bxg؇>  "          @�Q���33�8��@&ffA��CN@ ��33�Y��?��AxQ�CR#�                                    Bxgؕ�  �          @���Ǯ�\)@�\A���CI� �Ǯ�<(�?�AYCM=q                                    Bxgؤ�  �          @���ҏ\�G�@p�A��CG��ҏ\�-p�?��APQ�CJff                                    Bxgس0  �          @�\)����ٙ�?�{AffCA�������z�?333@��CCk�                                    Bxg���  
�          @���33��z�?fff@�Q�CB!H��33����?   @�{CCk�                                    Bxg��|  �          @�p����
@���]p���=qC�q���
@7
=�9�����C                                    Bxg��"  T          @߮���;�p��G��ָRC7u�����>�{�G���
=C0�H                                    Bxg���  �          @�ff���H��\)�@  ��  CB�f���H�h���R�\��
=C<z�                                    Bxg��n  
�          @�R��
=�n{���{C;�)��
=�\��R��{C7E                                    Bxg�  
�          @�{���
�E���(��
=C:n���
���R�ff���C6��                                    Bxg��  
�          @������{��\��  C>�����Q��!���p�C:xR                                    Bxg�(`  
�          @�{��=q��׿�  ���CE�3��=q��p���(��5CC�H                                    Bxg�7  �          @�
=��
=�R�\=���?@  CMB���
=�N�R�!G���ffCL�)                                    Bxg�E�  
�          @�(���33�S33�����z�CL����33�E������ffCKxR                                    Bxg�TR  T          @�
=�ƸR�#�
�(����G�CJaH�ƸR��z��H����  CE�                                    Bxg�b�  
�          @����z��2�\�e����CNQ���z��
=����
G�CF�H                                    Bxg�q�  T          @�������Dz��$z���33CM��������L(���{CI
=                                    BxgـD  T          @�����
�E���w�
CL#����
�"�\�-p�����CHO\                                    Bxgَ�  T          @�33��p��,(���\�W
=CI5���p���R�z���CEٚ                                    Bxgٝ�  
�          @���33?J=q�����l��C-�H��33?�����p��Qp�C*�q                                    Bxg٬6  �          @������>��������HC1z����?L���33�vffC-�)                                    Bxgٺ�  �          @�\)��녾���
=�6�\C7����녾����R�?
=C5
                                    Bxg�ɂ  �          @�Q��ᙚ�\)��ff�G
=C5��ᙚ>W
=��ff�F{C2O\                                    Bxg��(  �          @���
=�/\)�u����CJ���
=�p�����H  CH�
                                    Bxg���  T          @�����1G��O\)�θRCJ�f����!녿���333CI�                                    Bxg��t  �          @�G���  �p���Q��8  CF)��  ������r=qCC:�                                    Bxg�  �          @����Q�������s�CG�R��Q��\)�ff���HCD                                      Bxg��  
�          @������]p����H�5CPaH�����C�
�����RCM�                                    Bxg�!f  "          @�R��p��5��ff�'33CK�H��p��\)����t��CI5�                                    Bxg�0  
�          @�G��ָR�\)��(��ZffCF}q�ָR�����
�H��ffCC&f                                    Bxg�>�  �          @�33���
�
=��z��Xz�C8����
�aG���  �dQ�C5�{                                    Bxg�MX  �          @����=q�B�\�����(�C5����=q>�33�(�����C1�                                    Bxg�[�  T          @�G���G�>L���E���C2=q��G�?\(��>{���C,xR                                    Bxg�j�  �          @�  ��z�>�
=��
����C0����z�?\(����v�RC,�                                    Bxg�yJ  �          @ᙚ�߮=�Q�aG�����C3L��߮>�=q�W
=�ڏ\C1�                                    Bxgڇ�  
�          @�G����H?��� ����Q�C.�{���H?�=q����
=C)�3                                    Bxgږ�  
(          @��
����@33��(�W
B������@\(������j�B��                                    Bxgڥ<  
Z          @�(��\(�?����(���B�=q�\(�@L(���33z�B��                                    Bxgڳ�  T          @�(����?��
��\)u�C5����@����33�r��C�\                                    Bxg�  �          @陚��ff?����{�5p�C%.��ff@����G��$
=C0�                                    Bxg��.  �          @�
=��?�  ��{�'G�C)�{��@�\�����Cp�                                    Bxg���  
�          @�����@��{�CQ�C������@Mp����
�)=qC#�                                    Bxg��z  
�          @��H���?�{�u��
C#�����@	���]p����C�=                                    Bxg��   "          @�(��Ǯ@����
�p��C ���Ǯ@ �׿��\�+
=C{                                    Bxg��  "          @��H��=q@,(��ff��  C���=q@HQ��(��b=qC��                                    Bxg�l  
�          @���  @@���
=q��
=C^���  @Y����(��@��Cp�                                    Bxg�)  
�          @���G�@��?�=qA��C!�\��G�?�p�?��AD(�C#��                                    Bxg�7�  T          @�����z�@/\)?��RA�C���z�@��@!�A�C ��                                    Bxg�F^  �          @����Q�@E@�A��RC����Q�@"�\@8��A�G�C�                                    Bxg�U  T          @�{����@A�@  A�33C�f����@�R@6ffA�=qC��                                    Bxg�c�  �          @�����@P  ?�\)Aip�C)����@1G�@!G�A���C�{                                    Bxg�rP  "          @���33@S�
?У�AL��Cz���33@8��@�\A���C�\                                    Bxgۀ�  T          @�33��{@h��?�
=A3�C�=��{@P��@
�HA�CB�                                    Bxgۏ�  T          @�{�θR@W�?�G�A��Cp��θR@A�?���At(�C��                                    Bxg۞B  T          @�z����
@p��?��AM��Cp����
@Tz�@��A�  Cz�                                    Bxg۬�  
Z          @�����z�@!�?�G�A?�
C+���z�@	��@ ��A��\C"�                                    Bxgۻ�  T          @���z�@J�H?�G�@��C� ��z�@8Q�?�z�ANffC�=                                    Bxg��4  T          @�����H@C33?�Q�A333C&f���H@+�@33A��RC�H                                    Bxg���  �          @�p���  ?�(�@ffA��
C%8R��  ?�
=@*=qA��C)��                                    Bxg��  �          @�z���Q�?��?�p�Ax��C*�\��Q�?0��@
�HA���C.^�                                    Bxg��&  �          @�=q����?�ff?�ffA%��C%h�����?�p�?�33AR�HC'��                                    Bxg��  T          @����
@j=q?�@��HC����
@]p�?�=qA+�
C�=                                    Bxg�r  T          @����
@�ff�aG���p�C^����
@���=��
?&ffC�{                                    Bxg�"  �          @�G����@�G��Tz�����C�����@��
=#�
>�{Cn                                    Bxg�0�  �          @��H��  @n{��
=�t��C:���  @�G������
�HC�                                    Bxg�?d  "          @�\��p�@]p���z����C����p�@\��>�ff@aG�C�=                                    Bxg�N
  
�          @�33�У�@P  ���xQ�C� �У�@L��?\)@�C�{                                    Bxg�\�  T          @������@��R?�A�C\���@w�@�A�\)CQ�                                    Bxg�kV  �          @�Q���
=@�G�?!G�@��RC���
=@tz�?��HA:=qCh�                                    Bxg�y�  
�          @�
=��{@2�\?��
A1G�Cٚ��{@p�?�A�
C��                                    Bxg܈�  
�          @������H@W�@FffAݮC
���H@)��@o\)Bz�CJ=                                    BxgܗH  
�          @�Q����R@��R?���A;�C�����R@u�@��A���CW
                                    Bxgܥ�  �          @أ����H@(Q�>#�
?�z�C�)���H@"�\?5@���CY�                                    Bxgܴ�  "          @��
��@?\)@(��A�Q�C#���@�@L��A�33Cc�                                    Bxg��:  T          @ٙ���@K�@�G�B{CǮ��@p�@��B7=qC.                                    Bxg���  
�          @�����
=@�33@\)A�C����
=@~�R@X��A�RC��                                    Bxg���  
�          @׮�}p�@��?�(�A-��C ff�}p�@�(�@\)A�p�C��                                    Bxg��,  �          @��H��z�@I��?���AM��Cu���z�@333@   A�G�Cs3                                    Bxg���  T          @�G��ۅ?E�?��A  C-���ۅ?�?�p�A"{C/�H                                    Bxg�x  �          @�=q��z�?0��?�ffA*=qC.B���z�>��?�A9��C0�H                                    Bxg�  �          @����׮>k�?�Q�A Q�C2��׮��?��HA#33C48R                                    Bxg�)�  �          @ᙚ��
=>��?5@�=qC0&f��
=>��R?J=q@ϮC1s3                                    Bxg�8j  T          @��H�ٙ�?z�H?У�AUC+Ǯ�ٙ�?�R?�ffAlQ�C.�=                                    Bxg�G  �          @�z����@e���
�0��C\)���@a�?+�@���C��                                    Bxg�U�  T          @�R��@���G���  B�(���@���>�G�@`  B���                                    Bxg�d\  �          @��J=q@θR�Ǯ�G
=B�33�J=q@���?u@�{B螸                                    Bxg�s  �          @�R���H@�녾����
=CQ����H@�\)?u@�{CǮ                                    Bxg݁�  
�          @�R��=q@�G�>�{@/\)CJ=��=q@��?�Q�A9�CQ�                                    BxgݐN  T          @�(���{@���>u?�C	=q��{@�z�?�p�A (�C
+�                                    Bxgݞ�  
�          @�33���H@�녾��
�%C�����H@���?#�
@�ffC�q                                    Bxgݭ�  
�          @ᙚ����@AG�?�{A;�
C�����@*�H?��HA�=qCJ=                                    Bxgݼ@  �          @�  �Ӆ?�
=?�@�Q�C%�Ӆ?\?fff@�33C'�                                    Bxg���  T          @޸R�=q@��\@��BB���=q@s�
@���BE��B���                                    Bxg�ٌ  
�          @��G�@��@s33B{B��H�G�@z�H@��RB-�C �                                    Bxg��2  "          @�  ��ff@��
@�\A�33C���ff@���@P��A��Ch�                                    Bxg���  "          @����{@vff?�Q�A��C���{@aG�?�(�A�
=CE                                    Bxg�~  �          @�����
@P��?E�@��C�3���
@A�?�
=A<��C��                                    Bxg�$  	�          @�{��p�?�@0  A�33C!G���p�?��@EA߅C'                                      Bxg�"�  T          @�G���  ?��@Tz�A��
C!޸��  ?�33@h��A�Q�C(��                                    Bxg�1p  "          @��
��z�@U@*�HA���Cc���z�@-p�@S�
A���CW
                                    Bxg�@  
�          @�R��\)@��?
=@�
=CE��\)@��
?��AS\)C�)                                    Bxg�N�  "          @�{����@�ff��(��^�RCJ=����@�p�?&ff@�  Ch�                                    Bxg�]b  T          @�33��G�@>�R?B�\@�C�R��G�@0��?�{A9p�Cp�                                    Bxg�l  
�          @�Q���
=�@  �\)����CM� ��
=�:=q�B�\����CM�                                    Bxg�z�  
�          @�ff���\��z�}p���HC_ٚ���\���\��
=���RC]�f                                    BxgމT  
�          @߮��33�����-p�����C4�H��33>���*�H��{C/��                                    Bxgޗ�  T          @�\)��  �xQ쿐����C<&f��  �5��ff�-C:�                                    Bxgަ�  �          @�33�,��@љ�<#�
=#�
B��,��@��?�{A1��B��                                    Bxg޵F  
�          @�\�`  @�Q��ff�o33B� �`  @�Q�����B�aH                                    Bxg���  �          @�=q��33@��׿�{�Up�C
k���33@�Q�#�
����C�R                                    Bxg�Ғ  �          @����z�@8�ÿУ��W�C���z�@J�H�}p���\C�R                                    Bxg��8  T          @��H��(�?�����C*��(�?�=q��ff�l��C&��                                    Bxg���  �          @����ff@
�H�����  C ����ff@(Q������G�C                                      Bxg���  "          @�\�Ϯ?��  ��C'�3�Ϯ?��Ϳ�z��|��C$
                                    Bxg�*  "          @�  �љ��������{�C8��љ���\)��33���C4��                                    Bxg��  
�          @���ָR?�  �����{C)z��ָR?�p������\)C%�\                                    Bxg�*v  
�          @�����H?���7
=��Q�C(���H?�\)�!G���C$!H                                    Bxg�9  �          @���(�?��\��\���\C+�=��(�?����
�c�C(O\                                    Bxg�G�  T          @��H��p������z���  C4�)��p�>\��\��{C0޸                                    Bxg�Vh  "          @�ff��(�@�Ϳ޸R�]C����(�@0�׿����\)Cc�                                    Bxg�e  �          @�{����?����p����HC&�R����@G���=q�d��C#�                                    Bxg�s�  �          @�
=��33<#�
�7���  C3����33?(��3�
����C.�                                    Bxg߂Z  �          @�  ��z�>�
=�{���C0����z�?aG�����33C-�                                    Bxgߑ   T          @��H���H>�ff��  �V�RC0xR���H?O\)��\)�F=qC-��                                    Bxgߟ�  u          @�=q����=��Ϳ��
�[�C3=q����>���(��T(�C0\)                                    Bxg߮L  
�          @�33��{���=q���C5\��{>�p��Q���Q�C1\                                    Bxg߼�  
�          @����
����\���RC5����
>�33�G���33C1.                                    Bxg�˘  
�          @�G���녿}p����p�C<���녾��H�   ���C7�                                    Bxg��>  T          @����H�W
=� ������C:����H�����(Q����\C6p�                                    Bxg���  
�          @����R�n{����{C;O\��R��
=�{��
=C7W
                                    Bxg���  �          @����H�{��33�H��CEc����H��=q���}��CBxR                                    Bxg�0  
�          @����޸R�����B{CA�����׿�33�k\)C>�3                                    Bxg��  
�          @����33�#�
�   �|Q�C4\��33>�녿��H�v�HC0�3                                    Bxg�#|  
�          @�������@@  �U����
C\����@hQ��(�����RCn                                    Bxg�2"  
j          @���@S�
�`�����C�q��@~{�/\)����C
=                                    Bxg�@�  
�          @��H���\@xQ��XQ����HC8R���\@�  �   ���
C	�                                    Bxg�On  
�          @�\��{@�R��\)�
=C����{@Dz��k����C�{                                    Bxg�^  "          @�\��ff?��R�(������C"����ff@\)�
�H��\)C�H                                    Bxg�l�  "          @�z���33@A��1G����C����33@b�\����p�C�H                                    Bxg�{`  T          @�R���R@5�vff��
=C����R@e��J�H��C�H                                    Bxg��  
(          @���p�@9���Tz��ծC�)��p�@a��(�����C(�                                    Bxg���  
(          @�
=����@'��|���p�C&f����@XQ��S�
�ծC�                                    Bxg�R  
          @�
=��p�@QG��#33��Q�C���p�@n�R���aG�C�
                                    Bxg��  
j          @����ff@qG����R�|  C�)��ff@����33�G�Cc�                                    Bxg�Ğ  T          @�p���(�@s�
�:=q��p�C�3��(�@��\�33����Cs3                                    Bxg��D  
�          @�ff����@@  �y�����HCp�����@p  �L(���ffC�                                    Bxg���  "          @�ff����@�R�z=q��
=C������@O\)�S�
��(�C�q                                    Bxg��  �          @�p����?�{�i����{C#.���@%��J�H��G�C@                                     Bxg��6  �          @�(���?�\)�5���Q�C(p���?�
=��R���C#�f                                    Bxg��  "          @�R�߮?�=q�G��}C)8R�߮?�(��ٙ��TQ�C&#�                                    Bxg��  
�          @�Q�����?��
�Tz��ҸRC&�
����@(��:�H��(�C!xR                                    Bxg�+(  �          @����H?����@  ����C&Q����H@���&ff��  C!�\                                    Bxg�9�  "          @�\)��=q?����h����C%����=q@��N�R�̸RC��                                    Bxg�Ht  
�          @�ff�ҏ\?�G��E���HC'
�ҏ\@Q��-p���
=C")                                    Bxg�W  T          @�\)����?Ǯ��33�
=C%������@���k����CxR                                    Bxg�e�  
          @�����@�\������Cu���@<�������\)C��                                    Bxg�tf  
�          @����dz�@QG������E�C	���dz�@�(����H�"33C#�                                    Bxg�  "          @������H@6ff��ff�=qC�����H@n{�p  ��C�                                    Bxgᑲ  
�          @������@G
=��Q��1{C������@�����\)��C�                                    Bxg�X  T          @�����
=@Q����0z�C#���
=@������
���CxR                                    Bxg��  �          @�z��,��@\)��z��J��B�#��,��@�����p�� �HB�ff                                    Bxgὤ  T          @�(���Q�@P  ��p��5Q�C����Q�@�=q�����C�=                                    Bxg��J  "          @����33@��p�� ffCk���33@Tz�����	��C+�                                    Bxg���  T          @�(���{@��z��*Q�C���{@HQ����H�p�C�                                    Bxg��  T          @��H��  ?�33������C#����  @&ff��ff��C��                                    Bxg��<  T          @�(����?�ff�����33C"0����@1G���G��	C�3                                    Bxg��  
�          @�33���
?�����(�� �RC&� ���
@�������C�                                    Bxg��  T          @����θR?����E���C$=q�θR@(��(����33CT{                                    Bxg�$.  
�          @�Q��߮@z`\)�(��C!���߮@#33�Y���У�C�                                    Bxg�2�  
�          @�G���p�?޸R���R�7\)C&\)��p�@G�����p�C$G�                                    Bxg�Az  �          @������H?�G��G�����C+�f���H?�(��   �w\)C(L�                                    Bxg�P   T          @�Q��љ��#�
�fff��(�C4��љ�?B�\�`�����\C-h�                                    Bxg�^�  "          @�����R>�33��p���HC0�����R?�p���  �ffC(W
                                    Bxg�ml  "          @�
=����?��
�����(�C*@ ����?��~{� ffC"G�                                    Bxg�|  
�          @�G����
?Ǯ�i�����C&0����
@�
�N�R���C 
                                    Bxg⊸  
�          @�Q�����?��\�Mp���33C).����?�
=�7���{C#�{                                    Bxg�^  �          @�  ��zᾊ=q��p���C6����z�?�R��(��G�C.33                                    Bxg�  �          @��H�\?˅���R�=qC%L��\@p��qG���33C�q                                    Bxgⶪ  �          @�\��\)?Y���W
=���HC,�\��\)?�ff�Fff��p�C'
=                                    Bxg��P  �          @��
��=q?����G�� (�C(G���=q@	���j=q��(�C!=q                                    Bxg���  "          @���=q?
=q���p�C/(���=q?��~{��C'\)                                    Bxg��  T          @����@#�
���'�C8R���@c�
��Q��G�C�=                                    Bxg��B  
�          @��H���@9����ff�C�����@u�}p����CL�                                    Bxg���  
�          @����Q�@3�
��33�\)C޸��Q�@o\)�xQ���=qC�H                                    Bxg��  
�          @�=q���@�
���R�CG����@J�H�g
=��
=Ch�                                    Bxg�4  T          @�33��=q?ٙ����33C$T{��=q@$z��n{��\)C�                                    Bxg�+�  T          @�33��
=@�����H��C����
=@U��mp���  C��                                    Bxg�:�  �          @����G�@��������RC���G�@H���Z�H�ՅC��                                    Bxg�I&  �          @����ʏ\?��w���C$�ʏ\@'
=�X������C��                                    Bxg�W�  
�          @�\��ff?������{C)!H��ff@��(��  C �                                    Bxg�fr  "          @��H��?����G���C)�{��@33��ff��C �3                                    Bxg�u  �          @���Ǯ?����w
=��\)C%�
�Ǯ@���Z�H�ظRC�3                                    Bxgヾ  �          @�G���33?z�H����G�C*u���33?��H�����
=C!z�                                    Bxg�d  �          @�G���
=?���z��33C(�3��
=@�������C ��                                    Bxg�
            @������?�z��n�R���C&�)���@(��U�����C )                                    Bxg㯰  �          @������@R�\?p��@�=qC\)���@@��?��ALz�CaH                                    Bxg�V  v          @�G���ff@���(��S33C#�H��ff@
=q=#�
>�z�C#O\                                    Bxg���  	�          @����@'�?L��@��
C�����@Q�?�{A%�C!B�                                    Bxg�ۢ  
8          @���  @1G�>u?�{Cp���  @)��?W
=@�p�CL�                                    Bxg��H  �          @�=q��\)@������
=C!���\)@&ff��ff�B{C��                                    Bxg���  
�          @�\��
=?�ff�����C#8R��
=@+��o\)��(�CǮ                                    Bxg��  
�          @�\����@7
=�����C�f����@l���]p���C&f                                    Bxg�:  "          @�\��  @U������   C����  @�(��L(�����C�                                    Bxg�$�  �          @�\��G�@XQ��b�\�޸RC����G�@��\�-p����C�{                                    Bxg�3�  �          @����H@��Tz���G�C
�3���H@��������
C5�                                    Bxg�B,  �          @�=q��Q�@���(Q����CxR��Q�@��H��z��K33Cp�                                    Bxg�P�  
�          @��H��\)@dz��=q��Q�C.��\)@�Q�Ǯ�>ffC5�                                    Bxg�_x  
(          @�=q��(�@E��$z���G�C:���(�@dz�����_�
C�                                    Bxg�n  
�          @����\)@c�
���}G�CG���\)@z�H������
C�
                                    Bxg�|�  �          @����=q@
=��
=�o�
C ���=q@.�R��{�((�C33                                    Bxg�j  
�          @�ff��{>L���%����C2T{��{?E��{���C-�                                    Bxg�  "          @�\����?   ����RC0�����?�G��
�H���C+�q                                    Bxg䨶  "          @�=q���?!G��L(���=qC.�3���?�=q�>{���C(��                                    Bxg�\  "          @������H?fff�`����z�C,0����H?�z��N{����C%�
                                    Bxg��  "          @���\)?�������C'n��\)@G��l�����C��                                    Bxg�Ԩ  
�          @�p��θR?Ǯ�tz����C&p��θR@��XQ���
=C�
                                    Bxg��N  T          @�����?���e��(�C'�q����@
=q�L(���
=C!��                                    Bxg���  �          @��
��\)?�\)�P  �ʏ\C(xR��\)@�
�7
=���HC"�                                    Bxg� �  �          @�����Q�@��p��[�C"�=��Q�@���(����C @                                     Bxg�@  "          @�R���H@�(��0����ffC	�{���H@��>�
=@U�C	ff                                    Bxg��  �          @�\)�ȣ�@K��z���
=C��ȣ�@dzῦff�#33C\)                                    Bxg�,�  f          @���=q@�
����l��C#8R��=q@���\)�+\)C h�                                    Bxg�;2  �          @�ff��
=@�p��1G���p�B��)��
=@��Ϳ�(��7�B�(�                                    Bxg�I�  �          @�z��E@��
�@  �¸RB��E@�z�����K
=B�{                                    Bxg�X~  �          @�ff��
=@(���R���C
=��
=@,(�������C�=                                    Bxg�g$  �          @�G�����������{CH#�����=q�<(���(�CCW
                                    Bxg�u�            @��������.�R����CFB������
�J=q���
C@��                                    Bxg�p  �          @��
����33��\��=qC?����W
=�#33����C:��                                    Bxg�  �          @��R��=q��z���{�C<����=q�&ff��\���RC9�                                    Bxg塼  �          @�Q�����33�	�����HC<ٚ���!G��
=���C8�)                                    Bxg�b  �          @��H��p���=q�p�����C<@ ��p��
=q�����
=C80�                                    Bxg�  �          @�z���G���\)��\�Q��C<}q��G��0�׿�p��j�HC9:�                                    Bxg�ͮ  �          @��\��  ��R�aG���{C8����  �\)��Q��'�C8)                                    Bxg��T  �          @������׽u��
=�
�\C4p�����>W
=���z�C2n                                    Bxg���  �          @�=q��  ?!G����aG�C/\)��  ?8Q쾣�
�
=C.�                                    Bxg���  �          @�33��{@U�@
=A�Q�C���{@,��@Dz�A�{C�                                    Bxg�F  �          @�����@?\)@�Av�HC�f��@�@-p�A�Q�C �f                                    Bxg��  �          @�(���\)?�=q�n{��Q�C,��\)?��\�&ff��  C*��                                    Bxg�%�  �          @�ff���R?���B�\��G�C(�����R?�
=�Ǯ�3�
C'�q                                    Bxg�48  �          @�
=���\?�\)�Q���z�C+�����\?��\�
=q�w
=C*                                    Bxg�B�  �          A Q����?��\��33�[�
C*�����?�
=��ff�3
=C'��                                    Bxg�Q�  �          Ap���  ?}p��5���ffC,xR��  ?�\)�!����C'��                                    Bxg�`*  �          Ap����?����*=q���C*����?�33�����C%��                                    Bxg�n�  �          A����=q?����
�o33C(z���=q?�p���33�>=qC%W
                                    Bxg�}v  �          A{����@@�׿�\)�33Cٚ����@P  �!G���CO\                                    Bxg�  �          A��\)@=q� ���f{C"#���\)@3�
��33��RCc�                                    Bxg��  �          A��
=?����{�TQ�C(����
=?�Q쿸Q��$��C%�                                    Bxg�h  �          A ����(�?������
�0  C%�f��(�@�׿��
��z�C#�                                    Bxg�  �          A���@0  ���qG�C ���@2�\>�?h��C��                                    Bxg�ƴ  �          A ����
=@
=�=p�����C$� ��
=@{�k���\)C#�3                                    Bxg��Z  �          A ������?�{����ffC*�����?�녿z�H���C({                                    Bxg��   �          @����ff=�G��   �h��C3(���ff?z���_33C/��                                    Bxg��  �          @���������3�
����C6\��>���2�\����C0�R                                    Bxg�L  �          @��\�Ӆ�����N�R���CDh��Ӆ�����fff���C=�                                    Bxg��  �          @���ڏ\����s�
��ffC9  �ڏ\>��
�u���C1O\                                    Bxg��  �          @�p���
=?��~�R��Q�C/0���
=?��R�n�R��C'xR                                    Bxg�->  �          A ����ff@z��������C����ff@X��������CJ=                                    Bxg�;�  �          A�\��
=@{��ff�=qC!���
=@N{�r�\��C�\                                    Bxg�J�  �          A33�ٙ�?�  �������\C'�
�ٙ�@{�s33�ݙ�C                                     Bxg�Y0  �          Ap���G�?�z����
� 33C!����G�@E������
Q�C�R                                    Bxg�g�  �          Ap���33@��z��z�C W
��33@P������p�C�H                                    Bxg�v|  �          @��R����?�33�������RC#�����@3�
�\������C��                                    Bxg�"  �          @�{�У�?�p�����\)C'0��У�@p��p����\C^�                                    Bxg��  �          A ����z�Y���z���Q�C:Q���zᾅ������ffC5��                                    Bxg�n  �          A Q���z�Q��Z=q����C:�\��z�<��
�`  ��
=C3��                                    Bxg�  �          AG����H��=q�{���{CA  ���H�z���ff����C8�H                                    Bxg翺  �          A z���
=�������H��{C@#���
=�����=q��RC7��                                    Bxg��`  �          @�
=�ָR��G��y�����HCB���ָR�B�\���R����C:z�                                    Bxg��  �          @�
=��녿�  �r�\��C>k���녾����~�R��z�C6z�                                    Bxg��  �          A ���˅?\)�����C.�R�˅?޸R���
�(�C$�q                                    Bxg��R  �          A���33=���(��33C2���33?�  ���R�ffC)O\                                    Bxg��  �          A Q��Ӆ�&ff���R��C9�{�Ӆ>���\)�Q�C0�                                    Bxg��  �          @�����
��ff�����33CC.���
�@  ��(��Q�C:u�                                    Bxg�&D  �          @���ƸR���
������C=ff�ƸR>L����Q��p�C2#�                                    Bxg�4�  �          @��R��{���������\)C@�
��{�.{����ffC5�=                                    Bxg�C�  �          @�ff���\��
�����0\)CI!H���\�#�
���
�>G�C:��                                    Bxg�R6  �          @����z��{��ff�,�CJh���z�Q����\�;�C<�)                                    Bxg�`�  �          @��R��녿�(���=q�2�CHQ���녿���z��?(�C9�q                                    Bxg�o�  �          @�
=��=q�Q����+�HCL{��=q�u���H�<��C>G�                                    Bxg�~(  �          @��
��33�\)�xQ����HCIp���33���H���
�z�C@�                                    Bxg��  �          @�
=��녿�����33��HCD� ��녿������ ��C9T{                                    Bxg�t  �          A (�������������$��C@�R���    ��{�+�C3�q                                    Bxg�  �          @�{��{������33�4{CA�H��{=�G���Q��:ffC2�=                                    Bxg��  �          @�����R��p����5�RC@�R���R>aG�����;  C1�                                    Bxg��f  T          @�\)�������(�� \)CC������{��(��*�C7Q�                                    Bxg��  �          @����  ����  ��\)CGL���  ������{��C>}q                                    Bxg��  �          A Q�����������CG@ ��녿��\���H��C=8R                                    Bxg��X  �          A���أ׿���z���33C?�)�أ׾�{���
��C6�)                                    Bxg��  �          A ����Q��   �g
=���CHB���Q��G���(���Q�C@�\                                    Bxg��  �          A Q����(������HCF����������33���C=n                                    Bxg�J  �          A Q����,(���z���CLh�����
=��p��!�CA�{                                    Bxg�-�  �          A (���Q��:�H��ff�=qCN����Q�����G��%��CC�H                                    Bxg�<�  T          A Q���z��z���  � �CG�q��z῕��ff���C>aH                                    Bxg�K<  �          A �����
������=q���CDn���
�\(����33C;^�                                    Bxg�Y�  h          @�\)��G���  �c33��Q�C<���G���\)�l(���33C4�=                                    Bxg�h�  �          A (������ ���mp��ݙ�CD�����ÿ�  ��33��G�C<c�                                    Bxg�w.  �          @�\)���ÿ�����\����CD���ÿE���p���\C:�                                     Bxg��  �          A (���\)�XQ�������RCR�=��\)��������CH��                                    Bxg�z  �          @����?h�ÿ�
=�*�HC-(���?��R��33�	G�C*�f                                    Bxg�   �          A Q�����?�z῔z���HC'�����?�33�333��G�C&�                                    Bxg��  �          @���  �}p��Q���  C;��  ��{�#33����C6��                                    Bxg��l  �          A z����0���:=q��ffC9Y���=��
�?\)��p�C3ff                                    Bxg��  �          A ����
=��Q��ff���\C@�q��
=���\�-p���{C;�\                                    Bxg�ݸ  �          A (����Ϳ�\)��
=�$(�CA�3���Ϳ�
=��{�V�HC>�)                                    Bxg��^  �          @�\)��{�����_�CC����{��G������{C?�                                     Bxg��  �          A (���33�5�<����CJ}q��33��(��e��CD�                                    Bxg�	�  �          A �����Ϳ�
=���~�HCB�H���Ϳ�ff�'
=��G�C=�                                    Bxg�P  �          Ap���p�?�ff�9�����C+����p�?�\�!G���ffC&�\                                    Bxg�&�  �          A{��=q?0���/\)���\C.����=q?����{��(�C)��                                    Bxg�5�  �          A����
�n{� ����33C:����
�u�*�H��z�C5�                                    Bxg�DB  �          A{��G�����G����C5{��G�>�ff��R�~�RC0�                                    Bxg�R�  �          A=q��=q�aG��
�H�xQ�C5����=q>�Q��
=q�v{C1aH                                    Bxg�a�  T          AG���G�?@  ���
�MG�C.����G�?�Q�\�/
=C+T{                                    Bxg�p4  �          Aff�   ?�ff>�  ?�G�C*� �   ?�Q�?z�@�(�C+�=                                    Bxg�~�  �          A���
=�L�Ϳ��xQ�C4\)��
==�Q���uC3O\                                    Bxgꍀ  �          A���?G��1����RC.#����?�p���R��z�C(��                                    Bxg�&  �          A33���?+���{�T  C/����?�녿�{�7�C+��                                    Bxg��  �          A�\���ͿxQ���H�Ap�C;  ���;���33�X(�C7}q                                    Bxg�r  �          A{�   ?@  �xQ���z�C.���   ?z�H�=p���\)C-                                    Bxg��  �          A�R�   >����
=�"=qC2���   ?������
C0B�                                    Bxg�־  �          A�R�{��=q���j�HC5��{���z�����C4޸                                    Bxg��d  �          A�H��<��
?B�\@�(�C3޸�녾B�\?=p�@�
=C5Q�                                    Bxg��
  �          A�H� (�?��H>�@N�RC)��� (�?��
?Tz�@��
C*�H                                    Bxg��  �          A=q� Q�k�?�  @�\C5��� Q���H?fff@��
C7��                                    Bxg�V  T          A��\)��<#�
=uCE�
��\)��\�!G����CD�q                                    Bxg��  T          A���p��(Q���
=�E  CQ^���p��Y����{�Z�RC>Q�                                    Bxg�.�  �          A(���  �>�R��33�4��CR�{��  �����{�M(�CBs3                                    Bxg�=H  �          A  �����7������-{CP�{���׿�p���
=�C��CA&f                                    Bxg�K�  �          A�
����+�����*��CNT{�������\�?33C?#�                                    Bxg�Z�  �          A���ff�Z�H���H�\)CT���ff��33���\�5{CG@                                     Bxg�i:  �          A
=�����.{������CME���׿��R����.(�C@#�                                    Bxg�w�  �          A
=�˅���R��
=�33CEY��˅��R���\�  C9�{                                    Bxg놆  �          A\)�׮��G���\)���C>�{�׮<��
�����
z�C3�)                                    Bxg�,  �          A\)��\)��=q�~�R���C<���\)=u���
��ffC3�=                                    Bxg��  �          A{��@{�
=q�xQ�C#� ��@�=�Q�?+�C#Q�                                    Bxg�x  �          A�\���?�G���
=�>=qC)����?�zῙ���33C&@                                     Bxg��  T          A{��  ?.{�33���C.�3��  ?���G��f�RC*��                                    Bxg���  �          A�H��
=�������33C<�
��
=�   �   ��
=C7��                                    Bxg��j  �          A
=���ÿ���8Q���33C=޸���þ��HQ���=qC7�                                     Bxg��  �          Aff���R�8Q������RC5L����R>��H����{C0\)                                    Bxg���  �          A�\��{?k���H���HC-+���{?Ǯ�z��lz�C(�=                                    Bxg�
\  �          A����R?#�
����p�C/:����R?���	���v=qC*u�                                    Bxg�  �          AG���ff>L���z���{C2����ff?W
=�
�H�z�\C-�                                    Bxg�'�  �          A z��θR��
=�q���(�CD���θR�J=q��p���C:��                                    Bxg�6N  �          A����
=�0  �7����
CI����
=���a�����CB�                                    Bxg�D�  �          A����p���p��I����p�CCh���p���G��dz���=qC<�                                    Bxg�S�  �          A����
=��\)�  ��z�CB
��
=��z��,(����RC<�)                                    Bxg�b@  �          A����  ?8Q��(�����\C.�=��  ?�
=�����
C).                                    Bxg�p�  �          A ������?�(��(���G�C(�q����@ff��
=�]C$p�                                    Bxg��  �          A z�����?�\)�ff��
=C%������@p���(��G�C!�
                                    Bxg�2  T          A���=q@:�H�\)��ffC�
��=q@`  �Ǯ�3
=C��                                    Bxg��  �          AG���p�@z��
�H�x��C"����p�@5���!C{                                    Bxg�~  �          A �����?�\)����(�C'�����@p���(��F�\C#�f                                    Bxg�$  �          A��������z���p�C4޸���?��������C/�                                    Bxg���  �          A���
?�z���H�c\)C)�=���
?�z´p��*ffC%�                                    Bxg��p  �          A����?Tz��7
=����C-�=��?�\)�   ���HC'�
                                    Bxg��  �          A����ff?�=q�E���Q�C)���ff@
=q�#�
����C#L�                                    Bxg���  �          A��{?�ff�)������C&n��{@�R�G��f�RC!��                                    Bxg�b  �          A���R?�p��J=q���RC&xR��R@$z��!���{C \)                                    Bxg�  �          Ap����@ ���4z���{C$����@/\)���t  CQ�                                    Bxg� �  �          A{��\@G
=������CQ���\@j�H��33��HC�)                                    Bxg�/T  �          A{���@�
�7�����C$G����@333�	���up�C�                                    Bxg�=�  �          A{��Q�@A��-p�����C�H��Q�@k����H�H��Cff                                    Bxg�L�  �          A�H���H@�  �:�H���C&f���H@�p��Ǯ�/�C.                                    Bxg�[F  �          A
=�Ӆ@e��N{���C���Ӆ@��
��\�f�RC�\                                    Bxg�i�  �          A�R���
@333�W
=����CǮ���
@j�H�������C�f                                    Bxg�x�  �          A���ff@��[���(�C!.��ff@QG��%�����C��                                    Bxg�8  �          A���33?��������RC(���33@%��\�����CY�                                    Bxg��  �          A���(�@G��~�R��Q�C#����(�@G
=�Mp���(�C�                                    Bxg���  �          A(���ff?���fff���HC&��ff@1G��:=q���C�                                    Bxg��*  �          A����33@333�J�H��z�Cs3��33@g�����v�HC�                                    Bxg���  �          Az���R@�
�Tz�����C"8R��R@L(��\)��{C)                                    Bxg��v  �          A����\)?�z��=p�����C%����\)@,���  �}G�C !H                                    Bxg��  �          A�����H?޸R�Z=q��Q�C&�����H@*�H�.�R��=qC                                       Bxg���  �          AG�����?�ff�(���(�C(Ǯ����@�Ϳ�{�O
=C$:�                                    Bxg��h  �          AG����H?�
=�3�
���C%����H@+��ff�j=qC �{                                    Bxg�  �          Ap���\)?˅�N{��z�C(���\)@�R�%���C!��                                    Bxg��  �          Az��ٙ�@xQ��3�
���
CO\�ٙ�@�G���G��'�
CB�                                    Bxg�(Z  T          AQ����
@����
=q�r�RC� ���
@�Q�Y�����
C�R                                    Bxg�7   �          AG���(�@aG���R��ffC�R��(�@�33��ff��
C{                                    Bxg�E�  �          A����@K��3�
���HC����@xQ��(��@  C\)                                    Bxg�TL  �          Az���@-p��N�R��=qC���@c�
����}�C\)                                    Bxg�b�  �          AG�����@.{�����P��C !H����@I���p�����HCB�                                    Bxg�q�  �          AG���R@!G��QG�����C ��R@X���
=���C��                                    Bxg�>  �          A���������G��  CHc����c�
������\C;�                                    Bxg��  �          Az������
=�����G�CF=q���Ϳ����{�C8�                                    Bxg  T          Az���ff������
��
CD!H��ff�\)���$��C5L�                                    Bxg�0  �          A����{�����(���CDO\��{������
=�
=C6��                                    Bxg��  T          AG������  ��ff�$z�C=.���?:�H����&
=C-J=                                    Bxg��|  �          Ap���\)�xQ���{�,��C=@ ��\)?Q���
=�-�C,(�                                    Bxg��"  �          A��Ϯ������p��  C6��Ϯ?��
�������C(޸                                    Bxg���  �          AG���\)��Q���  �(�C6�R��\)?�ff��z���G�C+xR                                    Bxg��n  �          A����z�>�����R��{C1���z�?�ff�z�H���C'�                                    Bxg�  �          A�����H=����r�\��=qC3:����H?��\�dz�����C*33                                    Bxg��  �          AG����>����  ���
C0:����?�G��y����33C%��                                    Bxg�!`  �          A���?�G��j=q�ѮC,.��@ff�J=q���C${                                    Bxg�0  �          AG���ff>\�hQ��υC1���ff?�  �U���G�C(��                                    Bxg�>�  �          A����{>�
=�c�
��Q�C0� ��{?\�P  ��p�C(}q                                    Bxg�MR  �          A���\?˅�e�̣�C'Ǯ��\@(Q��:=q��  C @                                     Bxg�[�  �          A����@\)�i����(�C"k����@QG��1G����RC#�                                    Bxg�j�  T          Az���
=@+��z=q���CB���
=@qG��7���\)C��                                    Bxg�yD  T          A���ٙ�@Dz��g����HC���ٙ�@����{��ffC5�                                    Bxg��  �          A����H@a��e��G�C�����H@�\)��\��Q�C�                                     Bxg  �          A(�����@G
=�)����=qC!H����@r�\���
�-��C��                                    Bxg�6  �          A����Q�@����e���C^���Q�@������S
=C}q                                    Bxg��  �          A���(�?�
=�[����C'0���(�@+��.{����C 
=                                    Bxg�  �          AG���R?�(��U���C(�)��R@p��,�����
C!�                                    Bxg��(  �          A����
=@HQ��i�����
C���
=@�(��{����Cp�                                    Bxg���  �          AG���ff@S33�j=q����C����ff@�G��=q����CY�                                    Bxg��t  �          AG����@QG��^�R���CW
���@��R�  �{33C@                                     Bxg��  �          A���Ӆ@c�
�b�\��  C���Ӆ@�Q��p��w33C��                                    Bxg��  �          AG���  @g��J=q��p�C��  @�ff��=q�L  C�)                                    Bxg�f  �          Ap��׮@r�\�Fff���HC�f�׮@��H��(��>{C                                    Bxg�)  �          Ap���ff@C�
�Z�H��CG���ff@\)�  �zffC�                                    Bxg�7�  T          A��\@*=q�`  �ƣ�Cff��\@i���������C�R                                    Bxg�FX  �          A{��R@	���Dz�����C#�f��R@A��p��u��C޸                                    Bxg�T�  �          A����  @,����  ��z�C+���  @w
=�9�����CB�                                    Bxg�c�  �          A=q���@@  ������HC�q���@��:=q��z�C�                                    Bxg�rJ  �          A���=q?���  ��33C$�=��=q@G��J=q��C.                                    Bxg���  �          A�R��(�@(��o\)���C!���(�@a��/\)���
C��                                    Bxg���  �          A=q���
@���G���33C"����
@Q��
=q�p  C                                    Bxg�<  �          A=q����@�
�R�\���C$p�����@AG�����p�C                                    Bxg��  �          A���R?��QG����
C%�{��R@8Q��p����\C�)                                    Bxg�  �          A=q��ff@{�5�����C!�f��ff@QG���{�N{CT{                                    Bxg��.  �          Ap���@Fff�-p����C�
��@s�
���
�)C5�                                    Bxg���  �          A��ff@����8����Q�C���ff@�  ��33��\C�f                                    Bxg��z  S          A=q��
=@�Q��p����p�C�3��
=@�  �33�c33C	\)                                    Bxg��   T          A{��33@\�������CxR��33@�z��1����
C�)                                    Bxg��  �          A�ָR@5��33����C��ָR@�G��:�H���C�                                    Bxg�l  �          Ap����@���qG����
C"�����@U��4z����RC�q                                    Bxg�"  �          Ap���  @W����H�{C�
��  @e����
��C��                                    Bxg�0�  �          AG���z�@�{>k�?��C33��z�@�33?��HA@(�C5�                                    Bxg�?^  �          A����
=@��R=u>��CaH��
=@�?�ffA,��C�                                    Bxg�N  �          AG���@�=q>��R@��C����@}p�?޸RAAG�C޸                                    Bxg�\�  �          A�����@��
��
=�:�HC0�����@��?�z�@��C�                                    Bxg�kP  �          A�����@������Q�C�����@��\?u@��
CQ�                                    Bxg�y�  �          Aff��=q@^�R�J=q��{CO\��=q@b�\>�(�@;�C�                                    Bxg�  �          A
=���
@�{>�ff@K�C�R���
@�Q�@ ��Ae�CE                                    Bxg�B  T          A  ��ff@�
=>�@O\)C�
��ff@���@�\AeCh�                                    Bxg��  �          A�����H@n�R?G�@�p�C
=���H@P  @   A`(�C#�                                    Bxg�  �          AQ���R@`  >�p�@%�C�f��R@J=q?��A,  C�                                    Bxg��4  �          A����p�@�  ?�\)@��RC����p�@vff@%A���C��                                    Bxg���  �          A�R��ff@���>���@\)C&f��ff@{�?��
AC�CaH                                    Bxg���  �          A\)��{@]p�>\@'
=C��{@G�?�ffA(��C��                                    Bxg��&  �          A�� ��@��333����C#��� ��@p�>��?��
C#�                                    Bxg���  �          A33�
=@33�#�
��Q�C%��
=?�
=?0��@��RC&�q                                    Bxg�r  �          A33��R@Q�<#�
=�Q�C%\)��R@   ?@  @��\C&@                                     Bxg�  �          A���@
=<��
=���C#����@p�?Tz�@��
C$Ǯ                                    Bxg�)�  �          A�\��@
=q>��H@Z=qC$�=��?���?���A�\C&�
                                    Bxg�8d  T          A=q��R?Ǯ?k�@ə�C)0���R?�33?�33A=qC,�                                    Bxg�G
  
�          A���Q�?�R?+�@��
C/���Q�>�33?W
=@��C1�
                                    Bxg�U�  
�          A���33>k�?�G�@�  C2h��33�\)?��\@��
C4��                                    Bxg�dV  �          A���>.{�B�\��\)C2����>�(��&ff���C1�                                    Bxg�r�  �          A{��;B�\���\��Q�C5Q����>8Q쿂�\����C2Ǯ                                    Bxg�  �          A���Q쾙�����\���C6��Q�=��
�����G�C3u�                                    Bxg�H  �          A��
=?\)���
�(��C0��
=?����p��  C,�
                                    Bxg��  T          A��\)���
�Ǯ�,��C4��\)?���(��"=qC0J=                                    Bxg�  �          A
=�p���\)��  ���C4}q�p�>�p�������RC1n                                    Bxg�:  �          A����
��  =�G�?@  C:���
�z�H��  �޸RC:                                    Bxg���  
�          A=q���R�!G���R�yp�C8�����R>L����
��G�C2��                                    Bxg�ن  �          A���{@R�\?�(�A��C����{@(��@�
A�  C(�                                    Bxg��,  �          A\)��{������  ��\C>aH��{��  �����$��C;5�                                    Bxg���  T          AQ�� z�W
=�u��ffC5��� z�8Q���uC5@                                     Bxg�x  
�          A���녿�
=>���@3�
C>��녿�p����h��C>Y�                                    Bxg�  
�          A�����>u?�z�CAJ=���׾��H��CA                                    Bxg�"�  �          A������Q�@ ��A�ffCEz�����G�?��RA%p�CJk�                                    Bxg�1j  �          AQ�����j=q@A��
CO�������  ?aG�@��
CS33                                    Bxg�@  �          AQ���33�y��@�\A��CQ����33���R?=p�@�CU\                                    Bxg�N�  �          A���
=���@��A�CSs3��
=����?!G�@��CV�f                                    Bxg�]\            A��� z´p����H�[�C>n� zῙ���}p���{C<��                                    Bxg�l            A�R��H��z�k����HC=��H�z�H��\)��RC:�
                                    Bxg�z�            A
=��\��=q�����
=C>�3��\�xQ��{�K�
C:�\                                    Bxg�N  �          Aff�p���zῃ�
�ᙚCA@ �p���z��33�5p�C=ٚ                                    Bxg��  �          A�����z��\)��
=C6
��?!G������C/�\                                    Bxg�  �          AQ�� Q�>�ff�333���C0Ǯ� Q�?�
=�(����C)�)                                    Bxg�@  �          A(���R��z���\�zffC6���R?\)�\)�t��C0\                                    Bxg���  �          Az�����
=�p��pQ�C6�f��>�
=�p��pQ�C1)                                    Bxg�Ҍ  �          A���\)����������C6{�\)?
=��}��C/�)                                    Bxg��2  �          A	����B�\�33�]G�C5T{��?녿�p��Up�C0
                                    Bxg���  �          A������R�����G�C8:���=��Ϳ�Q��R=qC3L�                                    Bxg��~  �          A	��{���\��
=�ffC:��{��
=���H�8  C6޸                                    Bxg�$  �          A	����
�B�\�����\)C9#���
���ÿ�  ��C633                                    Bxg��  h          A����H�Tz῎{��RC9����H��p�����(�C6}q                                    Bxg�*p  �          A	G����?���7
=��C)�\���@ff����m��C#!H                                    Bxg�9  �          A�������녿��H��RC6�\���>#�
��G��$  C2��                                    Bxg�G�  �          A	��\)�������+
=C48R�\)?\)��  � (�C05�                                    Bxg�Vb  �          A	���
�L�Ϳ�Q��C4\)��
>��������C0�q                                    Bxg�e  �          A	G��{>�=q��p��:�\C2(��{?c�
��G��"=qC-�                                    Bxg�s�  �          A�
��\    �����(�C4���\>��Ϳ�G��ۅC1E                                    Bxg�T  �          A���ff��Q�����)��C4���ff?   ���R� ��C0�\                                    Bxg���  T          A33�	G��B�\�����C5=q�	G�>�33��{�Q�C1�                                    Bxg���  �          A33�	��#�
����=qC8B��	���Q쿽p��G�C4��                                    Bxg��F  �          A\)��Ϳ#�
���R���C8B���ͼ���\)�+�C48R                                    Bxg���  �          A��
=��z����X��C5��
=?   ���R�T(�C0��                                    Bxg�˒  �          AQ���
?
=�ff�]p�C0���
?����(��5�C+\                                    Bxg��8  �          Az��Q�>�=q�G��U��C2.�Q�?�G���\�:�RC-:�                                    Bxg���  �          A���
�R������33C7��
�R�#�
��33�ffC4
=                                    Bxg���  �          A  �{��
=����
�\C@�3�{�����Q��N=qC<�                                     Bxg�*  |          A��?�(��%���C(:���@$z��  �7�C"��                                    Bxg��  �          Ap��  >���\�pz�C0���  ?����Q��J�HC+E                                    Bxg�#v  �          A�� Q�?h���%�����C-� � Q�?�����\�^ffC'0�                                    Bxg�2  �          A	��
=>k�����(�C2k��
=?����
=q�jffC,�                                    Bxg�@�  �          A	���׿�G���ff�B�HC:����׾�  �33�^ffC5�R                                    Bxg�Oh  T          A
{�
=�Ǯ�Ǯ�'�C6�f�
=>k��˅�*�HC2h�                                    Bxg�^  �          A
ff�(��W
=��z����C9���(���{��33��C6B�                                    Bxg�l�  h          A
�R�33��=q�������
C>�{�33��ff�����)�C;\                                    Bxg�{Z  �          A���׿c�
��p����C9����׾�����H�5�C5��                                    Bxg��   �          A�
�	녿J=q�����Q�C933�	녾�z΅{��HC5�f                                    Bxg���  �          AQ��
ff��z῱���C5��
ff>����33��\C2@                                     Bxg��L  �          A
�R�	p�=u������Q�C3���	p�>��H���\��G�C0�q                                    Bxg���  T          A
�H�	��    ������G�C3���	��>�(���G��ָRC1&f                                    Bxg�Ę  �          A33�
ff����:�H��G�C5�q�
ff=#�
�E����HC3�q                                    Bxg��>  �          A\)�
ff>�{�\(����C1���
ff?&ff�(�����HC/��                                    Bxg���  �          A
�\��(�@Tz���
��z�Cff��(�@}p��W
=��
=C+�                                    Bxg���  �          A  ��
=@�Q����\��G�C����
=@�ff���>�\C��                                    Bxg��0  �          A��P��@��p����p�B�B��P��A Q�333��33B�B�                                    Bxg��  �          Az���33@���������C����33@�
=��\�y��C
!H                                    Bxg�|  �          A����
=@Y����  ��
=C���
=@�z��4z����C�                                    Bxg�+"  �          Ap�����@��������p�C�f����@��Ϳ���MG�Cu�                                    Bxg�9�  �          A=q��z�@�G���33���C�f��z�@��
�8Q����\C	�=                                    Bxg�Hn  �          AG����@@  ��Q��G�C���@��x����(�C\)                                    Bxg�W  �          Ap���?޸R�ҏ\�Dp�C"=q��@����
�G�C\)                                    Bxg�e�  �          A��(�@_\)��p���\)CE��(�@�G��:�H��ffC�                                     Bxg�t`  
L          A����?�=q��33���C$n���@x����p�����C\)                                    Bxg��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxg���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxg��R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxg���            A Q��33@\��������\)C33�33@��R�1���\)Cٚ                                    Bxg���  �          A�����@��������C")����@�z����\��G�C^�                                    Bxg��D  �          A33��{@c33�#33���RC���{@��׿k����C��                                    Bxg���  �          A��33@B�\�E��G�C����33@��׿�\)�&�\C޸                                    Bxg��  �          A�R��(�@@���3�
���C)��(�@xQ쿰����\CǮ                                    Bxg��6  �          A����\@E�N{���C�{��\@��
��(��4(�CxR                                    Bxg��  �          A���޸R?�p���{�	33C$��޸R@r�\�n�R�ɅCh�                                    Bxg��  �          A���\@��Ϳ&ff����CJ=��\@��?���@�33C��                                    Bxg�$(  �          A\)�ə�@�녿�Q����C	O\�ə�@���?��\@�\)C�H                                    Bxg�2�  �          A
=��
=@�33�  �qC�
��
=@��
�W
=��z�C�R                                    Bxg�At  �          A{����@�p������ffC����@��þ��>�RC^�                                    Bxg�P  �          A����
=@U��.�R���
C����
=@��Ϳ�33��C�                                    Bxg�^�  �          A{��Q�?�G��s33��p�C)��Q�@:�H�7����\Ch�                                    Bxg�mf  �          A���ᙚ@�=q���R� ��C��ᙚ@���>��H@R�\C�)                                    Bxg�|  �          A�����
@�  ��33�,��C�)���
@�
=?�R@���Cz�                                    Bxg���  �          A�H��Q�@��'
=��  C����Q�@�����=q��G�Ch�                                    Bxg��X  �          A�\��ff@�ff��z��F=qC.��ff@���>\@�HC}q                                    Bxg���  �          A(���=q@����0  ��Q�C����=q@����^�R��G�C��                                    Bxg���  "          A����@���Q��[�C� ��@�
=>aG�?�
=Cp�                                    Bxg��J  �          Ap��ٙ�@���8����{C
�ٙ�@�33����_\)CT{                                    Bxg���  �          A�����
@���333��ffCJ=���
@�ff?���A��C&f                                    Bxg��  �          A����@��׿#�
���
Ch����@��H?�p�Ap�Cc�                                    Bxg��<  �          A�����@����   �Mp�C����@�{<�>W
=C�3                                    Bxg���              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxg��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxg�.             A�R��p�@�  �:=q���\C���p�@��H��  ��
=C�)                                   Bxg�+�  �          A\)��33@ff�����C!����33@��
�Y����  C޸                                   Bxg�:z  �          A���ٙ�?�\)���R��C&���ٙ�@u������RC�=                                   Bxg�I   �          A����\)?�G�������C'���\)@X���s33�Џ\C{                                   Bxg�W�  �          A��
=@�\���R���HC%����
=@g
=�>�R���RC��                                   Bxg�fl  �          A{��(�?�����
=�&C*���(�@k�������C��                                   Bxg�u  �          A����Q�@   ���R�ffC%{��Q�@��H��(���ffCh�                                   Bxg���  �          AG���@QG���p���C���@�p��8Q���(�CY�                                   Bxg��^  �          AG����@>{��33��RC0����@��
�<(����Ch�                                   Bxg��  T          AG���@o\)�E���RC�=��@�{���R��ffCO\                                   Bxg���  �          Ap��33@c�
�)������C#��33@�33�k���Q�C�=                                   Bxg��P  T          A�R��@P  �&ff�yp�C����@�G���  ��p�C&f                                   Bxg���  �          A��\)@=q�:�H����C$��\)@Z=q�����HC#�                                   Bxg�ۜ  �          A  ��H?��H�Vff���C'�H��H@L(��{�R�RC \)                                   Bxg��B  �          A33�33@��C�
��G�C&���33@Mp�����3\)C @                                    Bxg���  �          A\)�G�?����c�
��Q�C(T{�G�@I���p��k
=C W
                                   Bxg��  �          A  �{?�(��fff���C)��{@E��"�\�q��C ٚ                                   Bxg�4  �          A���R?��H�R�\��ffC'�)��R@J�H�	���L��C u�                                   Bxg�$�  �          A\)�
�R?�\)�\)�¸RC)c��
�R@J=q�;�����C�R                                   Bxg�3�  �          A�
���?�
=��\)��Q�C&�����@p  �^{����C��                                   Bxg�B&  �          A�\��\)@������C#
��\)@�Q��W���\)C��                                   Bxg�P�  �          A\)���?����\)�  C){���@g����\�Ǚ�C��                                   Bxg�_r  �          A�R���H?�  ��ff�C'����H@�Q�����ٮC.                                   Bxg�n  �          A  ��ff?��\�׮�7C+
=��ff@r�\��p����C�                                   Bxg�|�  �          A�
��?#�
���
�a�HC,�R��@|(���33�9�HC�q                                   Bxg��d  �          A����?�G����p�
C!�\����@�p���(��7\)C�{                                   Bxg��
  T          A���{?�G����rz�CaH��{@�
=����5  CaH                                   Bxg���  �          A����?E��	�z33C*����@�z���ff�H��C!H                                   Bxg��V  �          A�H���?У���
�v  CE���@��
��
=�8��C�                                   Bxg���  �          A���\)?�
=��\)�Rz�C&���\)@���˅�#�C33                                   Bxg�Ԣ  �          A���Q�?˅��\)�I{C%8R��Q�@�����=q�p�CE                                   Bxg��H  �          A�H���R?�=q��\�N�C"=q���R@�p������  CG�                                   Bxg���  �          A=q��z�@���=q�E��C +���z�@�=q����RCB�                                    Bxg� �  �          A33��ff@{���
�<�HC G���ff@��\���R�	
=C��                                    Bxg�:  �          AQ���{@ ������1  C#L���{@�  ������RC�)                                    Bxg��  �          A������?�  ���H�9z�C$�3����@�����(��Q�C��                                    Bxg�,�  �          A�
���>�������HC0�
���@%���\��\)C!O\                                    Bxg�;,  �          A���Q�?aG���\)�
C-J=��Q�@Dz����\����C�                                    Bxg�I�  �          A��	p����H�1����
C@޸�	p��\)�Vff��=qC7��                                    Bxg�Xx  �          A�H��@ff�������C#����@�����=q�У�C��                                    Bxg�g  �          A�
� (�?�������  C(33� (�@e�dz����HC�H                                    Bxg�u�  �          A�R� z�?��
�����{C*�3� z�@N{�mp�����C#�                                    Bxg��j  �          A�H�	�?�G��\(���=qC+���	�@(���"�\�{�
C#                                      Bxg��  �          Az���?��H�G���G�C*�H��@+��
�H�R�RC#+�                                    Bxg���  �          Az����@33�xQ�����C$�H���@p����R�rffC�q                                    Bxg��\  �          AQ����@i���z��l��Cs3���@������C33C��                                    Bxg��  �          Ap��(�?���,����p�C*G��(�@$z�޸R�(��C$�                                    Bxg�ͨ  �          A�R�{>B�\�?\)���HC2���{?�p��&ff�{
=C*                                    Bxg��N  �          Aff��H�n{�0  ����C9Ǯ��H>\�8Q���33C1�f                                    Bxg���  T          A=q���ٙ��z��I�C>�����(��&ff��C7�H                                    Bxg���  �          Az������ÿ��O\)CB�3�����z���
�C?�
                                    Bxg�@  �          A��R���H@   AG�C=Q���R���?�33@��
CA�{                                    Bxg��  
�          A  ��  @��
@�
=A�G�C�R��  @<��@��B:�
C�                                     Bxg�%�  T          A���z�@���@<��A�ffCff��z�@(��@��HB	�HC�                                    Bxg�42  �          AG��=q?�\)@�AS\)C)��=q>��H@*�HA�=qC0�                                    Bxg�B�  �          AQ��G��\)@g
=A�Q�CD�H�G��u�@Q�AP��CL�3                                    Bxg�Q~  �          Az���\�C33@#33Ar�RCF����\�w�?n{@��RCKxR                                    Bxg�`$  �          A�
��?���@�p�A���C*#����u@���B ��C:��                                    Bxg�n�  �          A(��'
=@#�
A{B��{Cz��'
=��=qA�RB���CO
=                                    Bxg�}p  �          Az����R?�p�@��
BF�RC ����R��z�@�  BK�
CA޸                                    Bxg��  �          AQ��p��\(�@��A��
C:��p��/\)@c�
A�33CF��                                    Bxg���  �          Az�����(�?��A7\)C;������
=?��@���C?ٚ                                    Bxg��b  �          A{���c�
?�ff@�G�C9\)�����\?
=q@N�RC;�)                                    Bxg��  �          Az��33>\?aG�@��C1Ǯ�33�u?s33@�33C4W
                                    Bxg�Ʈ  �          A�R�Q�?У�@�AK
=C*��Q�?   @'�A|��C0�H                                    Bxg��T  T          A
=��H?���@ffAiG�C+h���H>��@,��A���C3�                                    Bxg���  �          Aff���?�=q>�p�@�RC-\)���?L��?O\)@�\)C/�                                    Bxg��  �          Aff��?�����\)��33C,����?��þ��3�
C*W
                                    Bxg�F  �          A��ff?�33��{��C,���ff?�
=>aG�?��C,�H                                    Bxg��  �          A�
=?���?���@��C+�
�
=?0��?�ffAz�C/�                                    Bxg��  �          A��
?���>�ff@0��C+� ��
?��
?��
@ə�C-�f                                    Bxg�-8  �          A�\��
?�ff?���@�z�C,  ��
?&ff?˅A(�C/�q                                    Bxg�;�  �          A���?�33?��@���C,����?
=q?�(�A��C0�H                                    Bxg�J�  �          A�=q?+�?�A333C/Ǯ�=q�aG�?�A>{C5c�                                    Bxg�Y*  �          A�H��?0��?��@j=qC/Ǯ��>�{?Y��@�C1�f                                    Bxg�g�  �          A33��?�{������HC+����?��>��?���C+��                                    Bxg�vv  �          A�R�
=?�=q�\(���
=C(� �
=@G�=�\)>�
=C'��                                    Bxg��  �          A��\)?�(��p����ffC'�f�\)@(�=u>\C&�f                                    Bxg���  �          A=q���?�녿�  �\)C(5����@�����8��C%Y�                                    Bxg��h  �          A�\�=q@���\�HQ�C%���=q@=p��E���
=C!�)                                    Bxg��  �          A�
�ff@z��(��V�\C%\)�ff@Dz�c�
��(�C �R                                    Bxg���  �          A  �(�?��ÿ�G��ffC+���(�?�{�=p���ffC(��                                    Bxg��Z  �          A�R��>�ff��R�p  C1:���?.{��{�z�C/��                                    Bxg��   �          A{��?k���G����RC.L���?�녿0����  C+h�                                    Bxg��  �          A33�{>�(�=�G�?#�
C1\)�{>�{>�z�?��
C1�3                                    Bxg��L  �          A�R��>W
=?+�@��HC2�R�녽�?0��@�
=C4�q                                    Bxg��  �          A
=�G�?�\?�ff@�p�C0�H�G���?�@�z�C433                                    Bxg��  �          A���Q�@녾\)�\(�C'O\�Q�?�\)?O\)@�G�C(O\                                    Bxg�&>  �          A��  @0���%�����C"��  @h�ÿ�G���Q�C�{                                    Bxg�4�  �          A���33?�Q���ff�G�C%33��33@�Q��_\)��G�Cff                                    Bxg�C�  �          A33���
?������\��
C(  ���
@o\)�q��ď\C!H                                    Bxg�R0  �          A=q����?�������

=C%�R����@�  �n�R����C5�                                    Bxg�`�  �          A33���>���Q�����C2E���?�  �2�\��p�C(u�                                    Bxg�o|  �          A�H���5��{��Q�C8������G�������C4��                                    Bxg�~"  �          A�H����J=q>W
=?�ffC8�R����L�;8Q쿓33C9�                                    Bxg���  �          A�
�=q����>Ǯ@{CNY��=q�p  ���
��\CL�                                     Bxg��n  �          Ap������fff@;�A���CL����������?k�@�\)CQ�                                    Bxg��  �          Ap��33��ff@XQ�A��RC@  �33�J=q@	��AV�RCH�                                     Bxg���  �          Ap����
�\)@x��A�\)CC�H���
�r�\@��Au��CM��                                    Bxg��`  �          AG���p���33@�G�B�C>W
��p��e@s�
A��
CM
                                    Bxg��  �          AG�����6ff��{�
�HCFs3����
=�У��'
=CCs3                                    Bxg��  �          A{�ff�5@��A�ffC8���ff�*=q@aG�A���CF�                                    Bxg��R  �          Az���R>�(�@�  B�C0�)��R��@��HB\)CD
                                    Bxg��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bxg��   ?          A�\���<�@x��A��HC3���Ϳ��@\��A���C@33                                    Bxg�D  �          A����>���@h��A���C1Ǯ����@W
=A�
=C=z�                                    Bxg�-�  �          Az��Q�?s33@'
=A�C-���Q�\@0  A�33C6�                                     Bxg�<�  �          A��	p��J�H?(�@r�\CHE�	p��E��u��
=CG�                                     Bxg�K6  �          A�\��Ϳu?��AAp�C:�H��Ϳ�Q�?���@�\C?��                                    Bxg�Y�  �          A�R�Q�c�
@Q�A�
=C9�Q��z�@{As�
CB�{                                    Bxg�h�  �          A���\�L��@5A�33C5J=��\��G�@=qAn�\C=��                                    Bxg�w(  y          Aff�
=�B�\@1�A�C8޸�
=��p�@AM�C@��                                    Bxg���  
�          A{�33<�?��@أ�C3�{�33���H?xQ�@�G�C7\                                    Bxg��t  
�          A�R�p�?.{>���?�C/�
�p�>�?
=@e�C133                                    Bxg��  �          A����?�(�>L��?��
C'xR��?��?�{@��C)�                                    Bxg���  �          Aff�p�@G��Q��k�C'��p�@:=q�����  C!Ǯ                                    Bxg��f  �          A�\�  ?��ÿ�Q���Q�C-ff�  ?�  �
=q�Q�C*                                    Bxg��  �          A
=�{?��@  ����C0�\�{?O\)����\)C/
=                                    Bxg�ݲ  �          A���?.{>�
=@!�C/����>���?0��@��RC1��                                    Bxg��X  �          A�R�\)=���@333A���C3Y��\)���R@ ��Axz�C;�f                                    Bxg���  �          A=q��?���?\(�@�Q�C(����?�(�?У�A!�C,aH                                    Bxg�	�  
Z          A�=q?�{    �#�
C(��=q?У�?aG�@�C)޸                                    Bxg�J  
(          A��\?c�
����C.u���\?�z�:�H��  C+E                                    Bxg�&�  F          A����=�?z�@dz�C3:����.{?�@`��C5�                                   Bxg�5�  "          A�����>�
=��\�I��C1h����?(���  ���C0:�                                    Bxg�D<  �          Az��\)=�Q�(��r�\C3xR�\)>�p��   �E�C1�R                                    Bxg�R�  /          A���+���ff����C8.����
���R��\)C4��                                    Bxg�a�  
�          A��������(����C4�{��>�׿������C1
                                    Bxg�p.  	`          A���
���Ϳ0������C6}q��
���
�L����{C4!H                                    Bxg�~�  �          A���33?����{���C-n�33?��H���6ffC*��                                    Bxg��z  �          A��G�?���=q�2�RC(�=�G�@\)�8Q���z�C$�f                                    Bxg��   �          A��  @ff���
�-G�C%Y��  @:�H��(��'
=C"�                                    Bxg���  �          A
=�
=q@N{���R�C
=C�=�
=q@q녾L�Ϳ���CY�                                    Bxg��l  "          A\)�
=q@E��G��^�HC aH�
=q@r�\��\�G�CJ=                                    Bxg��  �          A�H�
=q@Z�H�����z�CaH�
=q@i��>�@4z�C
                                    Bxg�ָ  	�          A�R���@9����z��
�HC!�����@N�R>#�
?z�HC�)                                    Bxg��^  
�          A{�
=?��
���Mp�C)��
=?�>���@�C(�
                                    Bxg��  "          Aff���@���33��  C&xR���@p�=L��>���C$��                                    Bxh �  
�          Aff���?�33���� Q�C(8R���@�\�k���z�C%ٚ                                    Bxh P  �          A��\)@�����ffC%aH�\)@0  �\)�\(�C"��                                    Bxh �  "          A��(�@S�
��ff�2=qC�q�(�@qG�<�>.{C�                                    Bxh .�  �          AG����@-p���p��=qC"ٚ���@E<��
=���C �H                                    Bxh =B  �          A����@5�fff����C"!H���@:�H?
=@i��C!��                                    Bxh K�  
�          A��Q�@>{�������C!G��Q�@P  >��?˅C��                                    Bxh Z�  G          A��  @Dzῦff�   C ���  @Tz�>�33@��C=q                                    Bxh i4  �          A{��@@�׿�G����C �R��@W�>\)?Y��C��                                    Bxh w�  
�          A�
�p�@S33��(�����Cn�p�@^�R?�@W
=C^�                                    Bxh ��  
�          A(�� Q�@�
=�u��ffC�)� Q�@��
?��AG�Cff                                    Bxh �&  
�          A���=q@��
�;���=qC����=q@����(��+�CE                                    Bxh ��  
�          A����G�@�z��9����\)C  ��G�@��ÿ�\�J=qC(�                                    Bxh �r  
g          AG��(�@K��7����C��(�@��s33��(�C.                                    Bxh �  "          A���33@r�\�(��r=qC+��33@�  ��\)�޸RC8R                                    Bxh Ͼ  
�          A�\��G�@���k���z�C�R��G�@��H���Q�CǮ                                    Bxh �d  
Z          A���33@y���tz�����C����33@��׿�
=�Q�CB�                                    Bxh �
  !          A�H� ��@{��7����C�� ��@��\�z��c33C\                                    Bxh ��  H          A33�
=@j=q��
�J�\C���
=@�ff�#�
���
C�=                                    Bxh
V  
�          A33���@}p��33�H��C�����@��R>��?fffC��                                    Bxh�  �          A(��Q�@������{Cٚ�Q�@�G�?u@�=qCB�                                    Bxh'�  �          AG����@�����Q��
ffCG����@�p�?xQ�@��HC�H                                    Bxh6H  "          A�@�{�����33C��@�z�?L��@��C�                                    BxhD�  T          A��@��׿����/�C.��@��H?!G�@s33C�=                                    BxhS�  T          A{����@����K����C�����@Ϯ    <#�
C�                                    Bxhb:  T          A�����@����I������C����@�=q��{�G�CJ=                                    Bxhp�  �          A���z�@���N�R����C(���z�@�녿��\��C�3                                    Bxh�  T          A������@����z=q��Cn����@�(���z����C�                                    Bxh�,  T          A�\��\@dz���=q���C��\@�33�*=q���\C�{                                    Bxh��  z          A���G�@\(����\�
��C����G�@���J=q���\C                                    Bxh�x  
�          A����33@g����H��G�C�=��33@���)���y��CT{                                    Bxh�  �          A�����@�(��i����\)CW
���@��\�c�
���CB�                                    Bxh��  
�          A���  @�(��S�
��  Cc��  @�\)�W
=��C�H                                    Bxh�j  "          A���=q@qG��R�\���C�=�=q@����  ���
C�                                    Bxh�  	�          A\)�  @dz��Q��>ffC=q�  @��=�\)>�ffCxR                                    Bxh��  
�          A\)�
�\@���u��
=C���
�\@s�
?��A4z�CG�                                    Bxh\  
�          A���(�@_\)�}p����CE�(�@aG�?^�R@�  C
                                    Bxh  �          A{��
@��\��ff��C���
@�  ?k�@���C�                                    Bxh �  �          A
=��(�@�
=�7
=���Ck���(�@�=q?��@r�\C��                                    Bxh/N  �          A�\� ��@��.�R��ffC(�� ��@�ff�.{���C�                                    Bxh=�  
�          A����  @��R�
=q�S�Cc���  @�(�?z�@a�C@                                     BxhL�  �          A
=��  @�ff����ٙ�C�\��  @��?��HA;�
C�
                                    Bxh[@  
Z          AQ���R@�ff��G���{C@ ��R@�z�@��AP��C�R                                    Bxhi�  �          A�
��(�@��������C����(�@�=q@0  A���C�                                    Bxhx�  
�          A=q����@�녿���&�HC@ ����@�?��@�G�C��                                    Bxh�2  �          A���G�@����{�mp�C�=��G�@���?��@c33Cc�                                    Bxh��  �          A�R���@�
=�:�H����CE���@�{>B�\?�33C
�                                     Bxh�~  "          Aff��  @�=q�Z=q����C����  @�=q���R���HC��                                    Bxh�$  �          AG���Q�@�p��J=q��ffC���Q�@�>���@
=C��                                    Bxh��  �          A{��\)@��\��\)���HC����\)@���B�\���C�)                                    Bxh�p  T          A�����
@�\)��33��C�)���
@�ff�˅���C}q                                    Bxh�  
�          A��ə�@�������	ffC���ə�@ۅ�33�]p�C�\                                    Bxh��  �          A������@�p����H�#�RC������@���L(����RCQ�                                    Bxh�b  "          A33���\@�33��=q���HC����\@�Q�p����z�C�R                                    Bxh  T          A��љ�@��\�U���=qC
^��љ�@ָR<�>8Q�CT{                                    Bxh�  "          A=q��{@�z��s33���C
Ǯ��{@��þ���8Q�C�                                     Bxh(T  �          Ap���
=@�  >��?xQ�Cu���
=@���@`��A�\)C��                                    Bxh6�  "          A=q���R@�=q��{�
=B������R@�Q�@W�A�  CL�                                    BxhE�  
�          A�
��
=@�?�@�=qB����
=@�@�(�A�C��                                    BxhTF  
(          A���Q�@�(�?B�\@�z�B�����Q�@�  @�33A���C��                                    Bxhb�  �          AG��\@��=��
>��HC� �\@���@n�RA�
=C�                                    Bxhq�  �          A���G�@�(�?!G�@p  C�R��G�@�p�@w�A�z�Ck�                                    Bxh�8  �          AG�����@�{�Ǯ�
=CB�����@�ff@H��A��C}q                                    Bxh��  �          A�����\@�p�?�G�A+�B������\@���@��B��CE                                    Bxh��  �          A�
����@�G�?��A{C������@�ff@�=qA���C0�                                    Bxh�*  �          A�R��G�@�Q�<#�
=#�
C�
��G�@���@a�A��\C�q                                    Bxh��  �          A{����@�33���\� (�B�\)����@߮@%A�  B�{                                    Bxh�v  T          A�H��Q�@�  �R�\����B�.��Q�@�?��@`��B��{                                    Bxh�  �          A���e�@��R@@  A�=qCc��e�@�@��RBB33CO\                                    Bxh��  T          A
�H��z�@��H@�\A���B�aH��z�@�ff@��RB8��B���                                    Bxh�h  �          A
{�(Q�@���@�A���B��(Q�@�33@�\)B733B�=q                                    Bxh  �          A33�R�\@�ff�����LQ�B���R�\@�33@z�At(�B�p�                                    Bxh�  �          A���p�@�{�l�����HB�����p�AG�>�(�@'�B�
=                                    Bxh!Z  �          A�
��Q�@Å��\)��G�CY���Q�@����  ��  B�\)                                    Bxh0   �          A��Q�@�=q��ff��RCz���Q�@��ÿ�\)��C��                                    Bxh>�  �          A����
=@e��Q��ٮCY���
=@�  ��\)�9�C�                                    BxhML  �          A�����@����(��u��C#�����@���>��@7
=Cff                                    Bxh[�  �          A���  @�ff��z��
=qC\��  @���@0  A��
Cp�                                    Bxhj�  �          A(��љ�@��Ϳ�{�P��C���љ�@�ff?=p�@�C޸                                    Bxhy>  �          A ���Ǯ@��H�p�����C(��Ǯ@��=�\)?�\CT{                                    Bxh��  T          @��H�S33@�ff��ff�8��B�L��S33@��?У�Ah��B�#�                                    Bxh��  �          @�
=@�@׮    ��B�Q�@�@��@Y��A��B���                                    Bxh�0  �          @��R��33@����H���B͞���33@�?���A;�
B��H                                    Bxh��  �          A	G��=q@�(��Dz���B���=qA{?�(�A�RB�{                                    Bxh�|  "          A  �z�@�ff�Vff���
B֔{�z�A�?c�
@��B��H                                    Bxh�"  �          A(����@��Mp���=qB�����@�
=?xQ�@أ�B�=q                                    Bxh��  �          A�R���R@�����H(�B�8R���R@��@�RA��BɸR                                    Bxh�n  �          A33��33A33�Y����G�Bʮ��33@�  @[�A�ffB̽q                                    Bxh�  �          A
=�(Q�A=q������B��(Q�@�@G�A�p�B�{                                    Bxh�  �          AQ��[�A �þW
=��{B�.�[�@ᙚ@z=qA�  B���                                    Bxh`  �          A	p��`��@��ÿ5��\)B�\�`��@��@UA��B��)                                    Bxh)  �          A
ff��{@�
=>��@N�RB�  ��{@���@|(�A�\)B�.                                    Bxh7�  �          A��ڏ\@�z��'
=��z�Cٚ�ڏ\@�=q=���?0��C�                                    BxhFR  
�          A33����@����R�o\)C�q����@��?&ff@�G�Cs3                                    BxhT�  �          A	��\)@�=q�;�����C����\)@���>��R@C^�                                    Bxhc�  �          A33����@ə��p��l��C=q����@љ�?�ffA	��C#�                                    BxhrD  �          A
=���H@Ӆ�.{���B�p����H@�\?��\@�Q�B�                                    Bxh��  �          A���Q�@b�\��ff�
=Ch���Q�@�=q�p���(�Ck�                                    Bxh��  �          A���33@E��ff���
Cu���33@����{�}�C��                                    Bxh�6  �          A  ��ff@1G���
=�߮C�)��ff@����{�d��C��                                    Bxh��  �          A
=���@
=�e���z�C#O\���@vff����A�C�f                                    Bxh��  �          A\)�ə�@�\)��ff�U��C��ə�@��?��@�ffC�                                    Bxh�(  �          A�\��=q@X���	���o
=C.��=q@�Q���^�RCJ=                                    Bxh��  �          A�����Ϳ�p������HCB}q���Ϳ�\)���R�f=qC<Y�                                    Bxh�t  �          A(���p�@mp��X����z�C����p�@�p����
��
=C�=                                    Bxh�  �          A����?�\)����
=qC%�
���@k��A���ffC��                                    Bxh�  �          A������@�z���\)�  C
������@ȣ׿�p��$(�C{                                    Bxhf  �          A�R��
=@�H���R���
C 8R��
=@�
=�Q����C�)                                    Bxh"  �          AQ���p��&ff�����\)C9aH��p�?�  ��\)���C%��                                    Bxh0�  �          A
�\��(��r�\�tz���33CS����(������
=�  C@\                                    Bxh?X  �          A	G����\��p����H�S�CfT{���\�������R���C\Q�                                    BxhM�  �          @�����Q�?��\�����+p�C&s3��Q�@i���j�H��z�C8R                                    Bxh\�  �          @�p�>����ȣ׿�(��t��C�|)>�����Q����7�
C�\                                    BxhkJ  �          @���Q����
�8Q���p�C�쿘Q�������(��p�C��=                                    Bxhy�  �          @�G�@��\����
=��p�C�\)@��\�q��<(��ޣ�C��                                     Bxh��  
4          @�{=u�Ӆ������C�=q=u�����fff���C�H�                                    Bxh�<  T          @�(�?���ۅ?�
=A�RC���?����  �\)���
C�\                                    Bxh��  T          @陚>�ff���H?+�@���C�Ф>�ff��{�C33��=qC�                                      Bxh��  T          @�\)����(�>��?��\C���������X����33C��3                                    Bxh�.  
Z          @��?(����H?}p�A33C��=?(���(��)�����C��R                                    Bxh��  "          @�{@J=q���H?��\AG�C�O\@J=q�����{��\)C�޸                                    Bxh�z  
�          @�G�@�G����
?n{A=qC�c�@�G���\)��Q��\z�C�ٚ                                    Bxh�   �          @��R@u���Q�?���AY�C���@u����
�}p��33C��)                                    Bxh��  
�          @�@�{����?�z�A z�C���@�{��p���=q�[�
C�=q                                    Bxhl  "          @�\@��R���?333@���C�H@��R��p��
=��\)C��                                    Bxh  
�          @~{@G
=��>�?�C��q@G
=�
=q�������C��H                                    Bxh)�  T          @dz�@L(��Ǯ�L�ͿQ�C���@L(�����W
=�[33C��3                                    Bxh8^  
�          @:=q?��L��>��RA=qC�f?��Y�������p�C�n                                    BxhG  �          @S�
@�;B�\��=q��HC���@��?&ff������Q�Am��                                    BxhU�  "          @5�?Tz�?fff�	���q=qB<�H?Tz�?�\)�����(�B��                                    BxhdP  
�          ?У�?�녿B�\��G���(�C�3?�녾�(��E����C��
                                    Bxhr�  
�          @	��?\���׿+���z�C���?\�!G���z����C��                                    Bxh��  �          @J�H@{����=�\)?�G�C��@{���ͿaG���z�C��                                    Bxh�B  "          @1G������?�ffBR��C�h������ff?:�HA��
C���                                    Bxh��  �          @#�
��33>��
?��B!�\C*ff��33��?�=qB�CD�=                                    Bxh��  �          @]p��>�R?W
=?�(�A��HC$8R�>�R��?�Q�A��C4��                                    Bxh�4  T          @�{����?�@�A�{C�{����?z�@1G�A���C,��                                    Bxh��  �          @�����(�?���@%�A�
=C s3��(�>u@I��A�Q�C1Q�                                    Bxhـ  T          @�z���33?=p�@��B!  C+�3��33����@�(�B�CEff                                    Bxh�&  T          @�ff��ff?��R@VffA��
C����ff�#�
@y��B=qC4\                                    Bxh��  T          @�Q���(�@   @J=qA�\)C���(�?�@�  B  C.��                                    Bxhr  
�          @�{��(�?�@W
=A�{C$�q��(��.{@s�
A��HC5s3                                    Bxh  T          @��R��33?�\)?�ffA9��C'�
��33?��@
=qA�ffC/^�                                    Bxh"�  �          @�ff���
?�{@{A��RC*� ���
�B�\@�RA�{C5�f                                    Bxh1d  	�          @���g
=?�?���Aә�C+�
�g
=���?�=qAծC;xR                                    Bxh@
  "          @ڏ\��G�?�{@z�A��C*  ��G��u@#�
A�G�C6.                                    BxhN�  T          @����(�?�p�@5�A���C&����(��8Q�@K�A��C5��                                    Bxh]V  T          @�{��33?�@AG�A��RC%.��33<#�
@aG�AٮC3��                                    Bxhk�  �          @�{��?���@#�
A�p�C$=q��=u@AG�A��
C3\)                                    Bxhz�  �          @�G���  ?��@/\)A���C&����  ����@@��A�G�C7
                                    Bxh�H  �          @�(�����?�33@C33AڸRC(� ���ÿz�@Mp�A��C9��                                    Bxh��  "          @�  ��(�?B�\@6ffA��C*����(��O\)@5�A�z�C=�                                    Bxh��  "          @��H�b�\?�\@\)B��C+���b�\�W
=@��B�RCAff                                    Bxh�:  �          @�G��xQ�?��@(��BffC O\�xQ�#�
@>�RB��C6aH                                    Bxh��  �          @�p��c33?��H@��BH�C!��c33���\@�\)BG��CG��                                    Bxh҆  "          @���k�?���@�RA��C�H�k�<#�
@'
=B=qC3�\                                    Bxh�,  
Z          @����^{?�=q@z�A��
C
=�^{�L��@*�HBp�C4�{                                    Bxh��  �          @�p����Ϳ�=q@K�B�\)C`�Ϳ����#�
@�B  Cv��                                    Bxh�x  �          @�Q��=q�޸R@p  BN��CW�=q�Z�H@�A��Cj��                                    Bxh	  �          @������Ϳ�@qG�B33C9��������@=p�A�=qCL�                                    Bxh	�  �          @�Q���p���@g
=B
z�CN���p��x��?��A��
C[�\                                    Bxh	*j  �          @�33�\)�I��@@��A���CZY��\)��  ?uA�Cb�{                                    Bxh	9  
�          @�(��Y����H@}p�B-�
CW� �Y�����@	��A��Cfk�                                    Bxh	G�  "          @����33� ��@���B7�CP����33��@VffA�\)Cb��                                    Bxh	V\  T          @�33�#�
�#�
@�33B�  C=q�#�
�Y��@��
Bf�RC��                                    Bxh	e  
�          @�?�\@�
=@�z�B5�RB��=?�\?��R@���B�W
B{                                    Bxh	s�  �          @�p�@���@J�H@�ffB;Q�B  @��׽��
@ə�Be��C�n                                    Bxh	�N  �          @���@QG�@��R@�ffB'z�BV��@QG�?��
@�G�Bw
=A��                                    Bxh	��  �          @��@z=q@�@�
=B
��BHp�@z=q?���@���BY(�A���                                    Bxh	��  �          @�R@Tz�@N{@0��A��\B0��@Tz�?���@���BA33A��                                    Bxh	�@  
�          @�33@w�?�(�@�{B\��A���@w����
@��BW��C��)                                    Bxh	��  �          @�(�@\�Ϳ333@�(�Bg��C�K�@\���^�R@�
=B)ffC�^�                                    Bxh	ˌ  �          @�=q@�  ?�\)@��\B1p�A�z�@�  ��ff@�33B2=qC�P�                                    Bxh	�2  �          @�  ?h��@p��@�{BZQ�B���?h��>�G�@�33B���A��                                    Bxh	��  �          @���@tz�?��@��HBU��At(�@tz����@��
BIp�C�q                                    Bxh	�~  �          @��R@�33?��@w�A�z�@�
=@�33��(�@h��A�(�C���                                    Bxh
$  T          @�
=@�ff���
@��B+\)C��R@�ff�*=q@���B\)C�H                                    Bxh
�  "          @��
@�
=�z�H@��B-(�C���@�
=�a�@\)A�{C��
                                    Bxh
#p  �          @�@p  ��33@�Ba�HC�)@p  ���H@��RB�C��H                                    Bxh
2  �          A ��@�=q��G�@�  B��C�W
@�=q�aG�@[�A��C�o\                                    Bxh
@�  "          A33@��
��@�\)B33C��@��
�n�R@?\)A�{C�K�                                    Bxh
Ob  �          A�R@�p��L��@���B!=qC�H�@�p��Y��@��
A��C���                                    Bxh
^  �          A
=@��
>�=q@�G�BU=q@H��@��
�G�@�  B3p�C���                                    Bxh
l�  
Z          A{@���>B�\@�Q�Bd��@p�@����Z�H@��
B<�RC�n                                    Bxh
{T  �          A�@\�L��@��B��C�@\�&ff@~{A��C�l�                                    Bxh
��  �          A(�@ڏ\����@z�HA��C���@ڏ\�P��@(Q�A��C�@                                     Bxh
��  
�          A=q@�(�>k�@���BR�@*=q@�(��E@��
B0�HC�ٚ                                    Bxh
�F  �          Ap�@k�?�=q@�
=BqQ�A��H@k��Q�@�z�BlC�                                      Bxh
��  �          Aff@y��?��
@�z�Bh�A��H@y���   @��HBez�C�p�                                    Bxh
Ē  �          A=q@5�?#�
@���B�B�AJ�R@5��.�R@�=qB_�C��                                    Bxh
�8  
�          A=q@��\��z�@��BP�C���@��\�;�@�33B#��C�&f                                    Bxh
��  �          Az�@�\)����
=�<(�C���@�\)�W��   ���RC�)                                    Bxh
��  �          A��@����@33A`��C���@���G�����J�HC���                                    Bxh
�*  T          A	@�ff�H��@��\A��HC�t{@�ff���@�Ad��C��H                                    Bxh�  "          A
�\@���h��@S33A�=qC�]q@�����?xQ�@�\)C��)                                    Bxhv  
�          A�@���~{@p��A�C��@�����\?���@��C�.                                    Bxh+  �          A��@���%@���B5�HC��H@����@��\A�z�C��
                                    Bxh9�  
�          A��@���   @��HBBp�C�AH@������@�z�A�C��                                    BxhHh  �          A�H@����ff@�33B>��C�t{@�����z�@�
=A�z�C���                                    BxhW  �          A�
@�����@޸RBKz�C��=@����  @���B
=C��3                                    Bxhe�  T          A
=@�Q쾣�
@�G�BO�C�Z�@�Q��w�@�z�B$��C�u�                                    BxhtZ  T          A=q@|�Ϳ
=q@��Bz��C�
@|����p�@��
B<ffC��                                    